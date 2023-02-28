from typing import List

import torch


def detection_post_process(outs: List[torch.Tensor], onnx=False):
    outputs = []
    for o in outs:
        o = o.view((o.shape[0], -1, o.shape[-1]))
        if onnx:
            o = torch.cat([o[..., :4], o[..., 4:].sigmoid()], dim=-1)
        else:
            o[..., 4:].sigmoid_()
        outputs.append(o)
    return torch.cat(outputs, dim=1)
