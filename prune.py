import argparse

import torch
from thop import clever_format, profile

import tools
from config import cfg
from pruning.pruner import SlimmingPruner
import analysis

def prune(config):
    cfg_path = config.model.cfg_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpus = config.system.gpus

    model_fun = lambda: tools.build_model(cfg_path, device_ids=gpus, device=device)[0]
    sp = SlimmingPruner(model_fun, config)
    sp.prune()

    model = sp.model.module
    new_model = sp.new_model.module
    inputs = torch.randn(1, 3, 512, 512).to(device)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    flopsnew, paramsnew = profile(new_model, inputs=(inputs, ), verbose=False)
    flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
    print("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
    # analysis.compare_PSNR(model, new_model)
    # analysis.compare_model_weights(model, new_model)
    sp.test()
    # sp.finetune()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pruner configuration')
    parser.add_argument('--yaml', default='yamls/voc.yaml')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.yaml)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    prune(cfg)
