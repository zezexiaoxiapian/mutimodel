import torch

import tools
from config import get_device
from dataset.eval_dataset import EvalDataset
from eval.evaluator import Evaluator
from pruning import block as PB
from trainer import Trainer
from typing import List
from functools import reduce

CFG_NET_SEGMENT = '[net]\nchannels=3'

def renumerate(sequence, start=None):
    if start is None:
        start = len(sequence) - 1
    for elem in sequence[::-1]:
        yield start, elem
        start -= 1

def _reverse_search_index(blocks: List, cls):
    for i, b in renumerate(blocks):
        if isinstance(b, cls):
            return i
    raise RuntimeError('could not find a Conv2d block')

class Constrainer:
    def __init__(self):
        self.links = []

    def find(self, link):
        for l in self.links:
            for le in link:
                if le in l:
                    return l
        return None

    def add(self, link):
        target = self.find(link)
        if target is None:
            self.links.append(set(link))
        else:
            target.update(link)

    def __iter__(self):
        return iter(self.links)

class SlimmingPruner:
    def __init__(self, model_fun, cfg):
        self.cfg = cfg
        self._prune_weight = cfg.prune.weight
        self._prune_ratio = cfg.prune.ratio
        self._prune_divisor = cfg.prune.divisor
        self._new_cfg = cfg.prune.new_cfg
        self._num_workers = cfg.system.num_workers
        self._device = get_device(cfg.system.gpus)
        state_dict = torch.load(self._prune_weight, map_location=self._device)
        model = model_fun()
        new_model = model_fun()
        model.load_state_dict(state_dict['model'])
        new_model.load_state_dict(state_dict['model'])
        print('load weights from %s' % self._prune_weight)
        self.model = model
        self.new_model = new_model
        self.blocks = []
        self._pruned_weight = self._prune_weight.rsplit('.', 1)[0] + '-pruned.pt'

        # pylint: disable-msg=no-value-for-parameter
        self.block_map = {
            'convolutional': lambda *x: PB.Conv2d(*x).with_args(divisor=self._prune_divisor),
            'maxpool': PB.Pool,
            'avgpool': PB.Pool,
            'upsample': PB.Upsample,
            'yolo': PB.YOLO,
            'shortcut': PB.ShortCut,
            'scale_channels': PB.ScaleChannels,
            'route': PB.Route
        }

    def prune(self):
        constrainer = Constrainer()

        for i, layer in enumerate(self.new_model.module.module_list):
            if layer._type in {'convolutional', 'maxpool', 'avgpool', 'upsample', 'yolo'}:
                input_layers = [] if len(self.blocks) == 0 else [self.blocks[-1]]
            elif layer._type == 'shortcut':
                constrainer.add((
                    _reverse_search_index(self.blocks[:layer._from+1], PB.Conv2d),
                    _reverse_search_index(self.blocks[:i], PB.Conv2d),
                ))
                input_layers = [self.blocks[layer._from], self.blocks[-1]]
            elif layer._type == 'scale_channels':
                self.blocks[-1].constrain_layer = self.blocks[layer._from]
                input_layers = [self.blocks[layer._from], self.blocks[-1]]
            elif layer._type == 'route':
                input_layers = [self.blocks[li] for li in layer._layers]
            else:
                raise ValueError("unknown layer type '%s'" % layer._type)
            self.blocks.append(self.block_map[layer._type](i, input_layers, layer))

        # gather BN weights
        bns = []
        maxbn = []
        for b in self.blocks:
            if isinstance(b, PB.Conv2d) and b.bn_scale is not None:
                bns.extend(b.bn_scale.tolist())
                maxbn.append(b.bn_scale.max().item())

        bns = torch.Tensor(bns)
        sorted_bns = torch.sort(bns)[0]
        prune_limit = (sorted_bns == min(maxbn)).nonzero(as_tuple=False).item() / len(bns)
        print('prune limit: {}'.format(prune_limit))
        if self._prune_ratio > prune_limit:
            # raise AssertionError('prune ratio bigger than limit')
            # since we have tackle prune-out issue in conv2d block (see block.py)
            # we wont raise an error but a warning
            print('the layer reached prune limit will be cast to 16 channels.')

        thre_index = int(bns.shape[0] * self._prune_ratio)
        thre = sorted_bns[thre_index]
        print('bn threshold: {}'.format(thre))
        thre = thre.to(self._device)

        # handle ShortCut
        for link in constrainer:
            link = list(link)
            mask = reduce(
                lambda x, y: x | self.blocks[y].get_out_mask(thre),
                link[1:],
                self.blocks[link[0]].get_out_mask(thre)
            )
            for bi in link:
                self.blocks[bi].constrain_mask = mask

        pruned_bn = 0
        segments = [CFG_NET_SEGMENT]
        for b in self.blocks:
            pruned_num = b.prune(thre)
            pruned_bn += pruned_num
            print("({}) {}: {}/{} pruned".format(
                b.layer_id, b.layer_name, pruned_num, len(b.out_mask)
            ))
            segments.append(b.reflect())
        cfg_content = '\n\n'.join(segments)
        with open(self._new_cfg, 'w') as fw:
            fw.write(cfg_content)

        status = {
            'step': 0,
            'model': self.new_model.state_dict(),
            'cfg': cfg_content,
            'type': 'normal',
        }
        torch.save(status, self._pruned_weight)
        print("Slimming Pruner done")

    def test(self):
        eval_dataset = EvalDataset(self.cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=None, shuffle=False,
            num_workers=self._num_workers, pin_memory=True,
            collate_fn=lambda x: x,
        )
        evaluator = Evaluator(self.new_model, dataloader, self.cfg)
        self.new_model.cuda()
        self.new_model.eval()
        AP = evaluator.evaluate()
        # 打印
        tools.print_metric(AP)

    def finetune(self):
        trainer = Trainer(self.cfg)
        trainer.run_prune(self._pruned_weight)
