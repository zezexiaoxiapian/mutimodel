import argparse
import datetime
import math
import os
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

import tools
from config import cfg, fix_gpus, get_device
from dataset.train_dataset import TrainDataset, TrainEvalDataset, collate_batch
from eval.evaluator import Evaluator
from tools import AverageMeter, TicToc
from thop import clever_format, profile
from copy import deepcopy
from model.loss import DetectionLoss


class Trainer:

    def __init__(self, config):
        # metric
        self.AP = None
        # model
        self._cfg_path = config.model.cfg_path
        # train
        self._train_batch_size = config.train.batch_size
        self._scheduler_type = config.train.scheduler
        self._mile_stones = config.train.mile_stones
        self._gamma = config.train.gamma
        self._init_lr = config.train.learning_rate_init
        self._end_lr = config.train.learning_rate_end
        self._momentum = config.train.momentum
        self._weight_decay = config.train.weight_decay
        self._warmup_epochs = config.train.warmup_epochs
        self._max_epochs = config.train.max_epochs
        # weights
        self._backbone_weight = config.weight.backbone
        self._weights_dir = os.path.join(config.weight.dir, config.experiment_name)
        self._resume_weight = config.weight.resume
        self._clear_history = config.weight.clear_history
        self._weight_base_name = 'model'
        # eval
        self._eval_after = config.eval.after
        self._eval_batch_size = config.eval.batch_size
        # sparse
        self._sparse_train = config.sparse.switch
        self._sparse_ratio = config.sparse.ratio
        # prune
        self._prune_ratio = config.prune.ratio
        # quant
        self._quant_train = config.quant.switch
        self._quant_backend = config.quant.backend
        self._disable_observer_after = config.quant.disable_observer_after
        self._freeze_bn_after = config.quant.freeze_bn_after
        # system
        self._gpus = fix_gpus(config.system.gpus)
        self._num_workers = config.system.num_workers
        self._device = get_device(self._gpus)

        self.init_eopch = 0
        self.global_step = 0
        self.config = config

        self.dataload_tt = TicToc()
        self.model_tt = TicToc()
        self.epoch_tt = TicToc()

        # summary writer
        self.writer = SummaryWriter(logdir=os.path.join('runs', config.experiment_name))
        self.scaler = GradScaler()

        self.scheduler = {
            'cosine': self.scheduler_cosine,
            'step': self.scheduler_step,
        }[self._scheduler_type]

    def scheduler_cosine(self, steps: int):
        warmup_steps = self._warmup_epochs * self._steps_per_epoch
        max_steps = self._max_epochs * self._steps_per_epoch
        if steps < warmup_steps:
            lr = self._end_lr + steps / warmup_steps * (self._init_lr - self._end_lr)
            lr_bias = 0.1 - steps / warmup_steps * (0.1 - self._init_lr)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr_bias if i == 2 else lr
                if 'momentum' in param_group:
                    param_group['momentum'] = 0.9 + steps / warmup_steps * (self._momentum - 0.9)
        else:
            lr = self._end_lr + 0.5*(self._init_lr-self._end_lr) *\
                (1 + math.cos((steps-warmup_steps)/(max_steps-warmup_steps)*math.pi))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

    def scheduler_step(self, steps: int):
        warmup_steps = self._warmup_epochs * self._steps_per_epoch
        if steps < warmup_steps:
            lr = self._end_lr + steps / warmup_steps * (self._init_lr - self._end_lr)
        else:
            for i, m in enumerate(chain(self._mile_stones, [self._max_epochs])):
                if steps < m * self._steps_per_epoch:
                    lr = self._init_lr * self._gamma ** i
                    break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def init_cfg(self):
        with open(self._cfg_path, 'r') as fr:
            self.cfg = fr.read()
        self.writer.add_text('config', self.config.dump())
        self.writer.add_text('model_config', self.cfg)


    def init_dataset(self):
        train_dataset = TrainDataset(self.config, self._meta_info)
        eval_dataset = TrainEvalDataset(self.config, self._meta_info)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._train_batch_size,
            shuffle=False, num_workers=self._num_workers,
            pin_memory=True, collate_fn=collate_batch,
        )
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self._eval_batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=True,
            collate_fn=collate_batch,
        )
        self.writer.add_scalars(
            'dataset',
            {'train_images': train_dataset.length, 'eval_images': eval_dataset.length},
        )
        print(f'{train_dataset.length} images for train.')
        print(f'{eval_dataset.length} images for evaluate.')


    def init_model(self):
        if self._quant_train:
            print('quantization aware training')
        self.model, model_info = tools.build_model(
            self._cfg_path, self._resume_weight, self._backbone_weight,
            device=self._device, clear_history=self._clear_history,
            dataparallel=True or not self._quant_train, device_ids=self._gpus,
            qat=self._quant_train, backend=self._quant_backend
        )

        tools.bare_model(self.model).raw = True
        self.global_step = model_info.get('step', 0)
        if self._quant_train and self.global_step == 0:
            def reset_bn_stats(mod):
                if type(mod) in set([torch.nn.intrinsic.qat.ConvBnReLU2d, torch.nn.intrinsic.qat.ConvBn2d]):
                    mod.reset_running_stats()
            self.model.apply(reset_bn_stats)

        self._meta_info = tools.bare_model(self.model).meta_info()
        self.ema = tools.ModelEMA(self.model)

    def summary_model(self):
        sizes = (512, 512)
        inputs = torch.randn(1, 3, *sizes).to(self._device)
        model = deepcopy(self.model.module)
        flops, params = profile(model, inputs=(inputs, ), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print('{} cfg: {}, MACs: {}, params: {}'.format(sizes, self._cfg_path, flops, params))
        del inputs, model
    def init_evaluator(self):
        self.ema = tools.ModelEMA(self.model)
        self.evaluator = Evaluator(self.ema, self.eval_dataloader, self.config)
    def init_optimizer(self):
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self._init_lr, weight_decay=self._weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self._init_lr, weight_decay=self._weight_decay, momentum=0.937, nesterov=True)
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        self.optimizer = optim.Adam(pg0, lr=self._init_lr, betas=(self._momentum, 0.999))
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self._weight_decay})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

    def init_losses(self):
        self.criterion = DetectionLoss(self._meta_info)
        self.losses = {
            'loss': AverageMeter(),
            'xy_loss': AverageMeter(),
            'obj_loss': AverageMeter(),
            'cls_loss': AverageMeter(),
            'loss_per_branch': [AverageMeter() for _ in range(3)],
        }
    def eval(self):
        ap, loss_dict = self.evaluator.evaluate()
        self.AP = ap
        # 打印
        tools.print_metric(ap, verbose=False)
        tools.add_AP_to_summary_writer(self.writer, ap, self.global_step)
        print('val loss: xy: {xy_loss:.4f}, obj: {obj_loss:.4f}, cls: {cls_loss:.4f}, all: {loss:.4f}'.format(**loss_dict))
        return ap

    def _clear_ap(self):
        self.AP = None

    def save(self, epoch):
        base_name = self._weight_base_name
        model_name = f'{base_name}-{epoch}.pt' if self.AP is None\
            else f'{base_name}-{epoch}-{self.AP.AP:.4f}.pt'
        model_path = os.path.join(self._weights_dir, model_name)
        status = {
            'step': self.global_step,
            'AP': self.AP,
            'model': self.ema.ema.state_dict(),
            'cfg': self.cfg,
            'type': 'qat' if self._quant_train else 'normal',
            'backend': self._quant_backend if self._quant_train else 'none',
        }
        torch.save(status, model_path)

    def train_epoch(self, epoch):
        self.dataload_tt.tic()
        for data in self.train_dataloader:
            self.global_step += 1
            image, index, label = data
            self.dataload_tt.toc()
            lr = self.scheduler(self.global_step)
            self.writer.add_scalar('train/learning_rate', lr, self.global_step)

            self.model_tt.tic()
            with autocast():
                radar = image[:, 4, :, :].view(self._train_batch_size, -1)
                image = image[:, :4, :, :]
                pred = self.model(image, radar)
                losses_dict = self.criterion(pred, index, label)
            tools.add_losses_to_summary_writer(self.writer, losses_dict, self.global_step)
            self.scaler.scale(losses_dict['loss']).backward()

            assert 64 % self._train_batch_size == 0
            acc_step = 64 // self._train_batch_size
            if self.global_step % acc_step == 0:
                if self._sparse_train:
                    for m in self.bns:
                        m.weight.grad.data.add_(self._sparse_ratio * torch.sign(m.weight.data))

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)

            self.model_tt.toc()
            for name, loss in losses_dict.items():
                if isinstance(loss, torch.Tensor):
                    self.losses[name].update(loss.item())
                else:
                    for i, l in enumerate(loss):
                        self.losses[name][i].update(l.item())
            if self.global_step % self._loss_print_interval == 0:
                loss_values = {}
                for name, loss in self.losses.items():
                    if tools.is_sequence(loss):
                        loss_values.update(
                            {f'{name}_{i}': l.get_avg_reset() for i, l in enumerate(loss)}
                        )
                    else:
                        loss_values[name] = loss.get_avg_reset()
            self.dataload_tt.tic()

        self.train_dataloader.dataset.init_shuffle()
        if self._sparse_train:
            bn_vals = np.concatenate([m.weight.data.abs().clone().cpu().numpy() for m in self.bns])
            bn_vals.sort()
            bn_num = len(bn_vals)
            bn_indexes = [round(i/5*bn_num)-1 for i in range(1, 6)]
            print('sparse level: {}'.format(bn_vals[bn_indexes].tolist()))

        dataload_time = self.dataload_tt.sum_reset() / 1e9
        compute_time = self.model_tt.sum_reset() / 1e9
        self.writer.add_scalars(
            'train/time',
            {'dataload': dataload_time, 'compute': compute_time},
            self.global_step
        )
        self.writer.add_scalar('train/epoch', epoch, self.global_step)
        print('data load time: {:.3f}s, model train time: {:.3f}s'.format(
            dataload_time, compute_time))

    def train(self):
        # 每一轮训练
        for epoch in range(self.init_eopch, self._max_epochs):
            self.model.train()
            self._clear_ap()

            if self._quant_train:
                if epoch >= self._disable_observer_after:
                    # Freeze quantizer parameters
                    self.model.apply(torch.quantization.disable_observer)
                if epoch >= self._freeze_bn_after:
                    # Freeze batch norm mean and variance estimates
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            self.epoch_tt.tic()
            self.train_epoch(epoch)
            self.epoch_tt.toc()
            epoch_time = self.epoch_tt.sum_reset() / 1e9
            eta = datetime.timedelta(seconds=epoch_time * (self._max_epochs - epoch - 1))
            print('{:.3f}s per epoch, ~{} left.'.format(epoch_time, eta))
            if epoch >= self._eval_after:
                if self._quant_train:
                    self.evaluator.model = tools.quantized_model(self.ema.ema)
                self.ema.ema.eval()
                self.eval()
            self.save(epoch)

    def run(self):
        tools.ensure_dir(self._weights_dir)
        torch.backends.cudnn.benchmark = True
        self.init_cfg()
        self.init_model()
        self.init_dataset()
        self._steps_per_epoch = len(self.train_dataloader)
        self._loss_print_interval = max(self._steps_per_epoch // 10, 1)
        self.init_eopch = self.global_step // self._steps_per_epoch
        # self.summary_model()
        self.init_evaluator()
        self.init_optimizer()
        self.init_losses()
        if self._sparse_train:
            self.bns = tools.get_bn_layers(self.model)
        self.train()

    def run_prune(self, prune_weight: str):
        self._cfg_path = self.config.prune.new_cfg
        self._init_lr *= 0.2
        self._warmup_epochs = 0
        self._max_epochs = 20
        self._backbone_weight = ''
        self._resume_weight = prune_weight
        self._clear_history = True
        self._eval_after = 0
        self._sparse_train = False
        self._weight_base_name = f'pruned-{round(self._prune_ratio*100)}-model'
        self.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trainer configuration')
    parser.add_argument('--yaml', default='yamls/flir.yaml')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    Trainer(cfg).run()
