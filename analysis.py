import torch
from torch import nn

import tools
from dataset.eval_dataset import EvalDataset
from config import cfg
from tqdm import tqdm
from collections import defaultdict


def compute_error(x, y):
    if x.numel() != y.numel():
        return float('nan')
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return (20 * torch.log10(Ps/Pn)).item()

def register_output_logger(model: nn.Module):
    def hook(mod: nn.Module, input, output):
        logged = output.clone()
        if output.dtype in {torch.qint8, torch.quint8, torch.qint32}:
            logged = logged.dequantize()
        mod.register_buffer('_logger_output', logged)
        return output
    for mod in tools.bare_model(model).module_list:
        mod.register_forward_hook(hook)
    return model

def compare_model_outputs(float_model, q_model):
    register_output_logger(float_model)
    register_output_logger(q_model)

    with torch.no_grad():
        cfg.merge_from_file('yamls/voc.yaml')
        cfg.eval.batch_size = 16
        cfg.freeze()
        dataset = EvalDataset(cfg)
        for data, *_ in tqdm(dataset):
            data = data.to('cuda')
            float_model(data)
            q_model(data)

    act_compare_dict = defaultdict(tools.AverageMeter)
    for i, (fm, qm) in enumerate(zip(
        tools.bare_model(float_model).module_list,
        tools.bare_model(q_model).module_list
        )):
        err = compute_error(fm._logger_output, qm._logger_output)
        act_compare_dict[f'{i}-{fm._type}'].update(err)
    return act_compare_dict

def compare_PSNR(float_model, q_model):
    act_compare_dict = compare_model_outputs(float_model, q_model)
    for key in act_compare_dict:
        err = act_compare_dict[key].get_avg_reset()
        print(f'{key}:\t{err}')

def compare_model_weights(float_model, q_model):
    wt_compare_dict = {}
    for i, (fm, qm) in enumerate(zip(
        tools.bare_model(float_model).module_list,
        tools.bare_model(q_model).module_list
        )):
        for (fn, fw), (_, qw) in zip(fm.state_dict().items(), qm.state_dict().items()):
            if fw.dtype is not torch.float32 or qw.dtype is not torch.float32:
                continue
            err = compute_error(fw, qw)
            wt_compare_dict[f'{i}-{fm._type}-{fn}'] = err
    for key in wt_compare_dict:
        err = wt_compare_dict[key]
        print(f'{key}:\t{err}')

if __name__ == "__main__":
    fmodel = tools.build_model(
        'model/cfg/mobilenetv2-fpn.cfg',
        'weights/voc_mobilenetv2_fpn_qat5/model-79-0.4768.pt',
        dataparallel=False,
        qat=True,
        device='cpu',
    )[0]

    qmodel = tools.quantized_model(fmodel, inplace=False)
    fmodel.apply(torch.quantization.fake_quantize.disable_fake_quant)

    compare_PSNR(fmodel, qmodel)