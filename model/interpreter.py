import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from model.parser import Parser
from typing import IO, Union
import math
from model.utils import detection_post_process
from torch.cuda.amp import autocast


class AnyModel(nn.Module):

    def __init__(self, cfg: Union[str, IO], quant: bool = False, onnx: bool = False):
        super().__init__()
        self.quant = quant
        self.qstub = QuantStub()
        self.destub = DeQuantStub()

        if isinstance(cfg, str):
            cfg = open(cfg, 'r')
        self.module_list = nn.ModuleList(Parser(cfg).torch_layers(quant, onnx))
        cfg.close()

    def is_output(self, i, layer) -> bool:
        return False

    def forward(self, x):
        cache_outputs = []
        outputs = []
        for i, layer in enumerate(self.module_list):
            if self.quant and i == 0:
                x = self.qstub(x)
            layer_type = layer._type
            if layer_type in {'convolutional', 'fc', 'upsample', 'maxpool', 'avgpool', 'batchnorm'}:
                x = layer(x)
            elif layer_type in {'shortcut', 'scale_channels'}:
                x = layer(x, cache_outputs[layer._from])
            elif layer_type == 'route':
                x = layer([cache_outputs[li] for li in layer._layers])
            elif layer_type == 'yolo':
                if self.quant:
                    x = self.destub(x)
                x = (layer._stride, layer(x))
            else:
                raise ValueError('unknown layer type: %s' % layer_type)
            if self.is_output(i, layer):
                outputs.append(x)
            cache_outputs.append(x)
        num_outputs = len(outputs)
        if num_outputs == 0:
            outputs = cache_outputs[-1]
        elif num_outputs == 1:
            outputs = outputs[0]
        return outputs


class DetectionModel(AnyModel):

    def __init__(self, cfg: Union[str, IO], quant: bool = False, onnx: bool = False):
        super(DetectionModel, self).__init__(cfg, quant, onnx)
        self.onnx = onnx
        # 如果 raw True，输出 logit 几率，并且保持输出的形状，方便计算 loss
        self.raw = False
        self.initialize_biases()

    def is_output(self, i, layer) -> bool:
        return layer._type == 'yolo'

    def initialize_biases(self):
        # 找到每个在 yolo 层前的最后一个卷积层
        last_convs = []
        pre = None
        for layer in self.module_list:
            # init gamma of final bn to zero
            if layer._type == 'shortcut' and layer._raw['activation'] != 'linear' \
                    and pre._type == 'convolutional' and hasattr(pre, 'bn'):
                pre.bn.weight.data.zero_()
            if layer._type == 'yolo':
                last_convs.append((pre.conv, layer.opt['stride'], layer.opt['classes']))
            pre = layer
        for conv, stride, num_cls in last_convs:
            b = conv.bias.view(-1, 5 + num_cls)
            b[:, 4] += math.log(8 / (608 / stride) ** 2)  # obj (8 objects per 608 image)
            b[:, 5:] += math.log(0.6 / (num_cls - 0.99))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def meta_info(self):
        # return strides and yolo options
        opts = []
        for layer in self.module_list:
            if layer._type == 'yolo':
                opts.append((layer._stride, layer.opt))
        opts.sort(key=lambda x: x[0])
        return dict(opts)

    @autocast()
    def forward(self, x, r):
        s = int(math.sqrt(r.shape[-1]))
        r = r.view(-1, 1, s, s)
        x = torch.cat([x, r], dim=1)
        outputs = super(DetectionModel, self).forward(x)
        # 按 stride 升序排序 (8, 16, 32) = (小, 中, 大)
        outputs = [o[1] for o in sorted(outputs, key=lambda e: e[0])]
        if self.training or self.raw:
            return outputs
        return detection_post_process(outputs, self.onnx)


class ClassifierModel(AnyModel):
    pass


if __name__ == "__main__":
    Model = DetectionModel
    # print(Model('model/cfg/mobilenetv2-yolo.cfg').module_list)
    from thop import clever_format, profile

    model = Model('model/cfg/regnety-400m-fpn.cfg')
    inputs = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:{}, params: {}".format(flops, params))
