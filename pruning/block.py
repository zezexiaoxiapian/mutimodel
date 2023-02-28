from collections.abc import Sequence

import torch
from torch import nn
from copy import deepcopy


def _is_none_or_zeros(t):
    if t is None:
        return True
    return t.sum() == 0

def _ceil(n, div):
    return ((n + div - 1) // div) * div

class Baselayer:
    def __init__(self, layer_id: int, input_layers: list, modules: nn.Module, keep_out: bool=False):
        self.layer_id = layer_id
        self.layer_name = modules._type
        self.input_layers = input_layers
        self.modules = modules
        self.input_mask = None
        self.out_mask = None
        self.keep_out = keep_out
        self.const_channel_bias = None

    def with_args(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def prune(self, threshold) -> int:
        self.out_mask = self.input_layers[0].out_mask
        return 0

    def reflect(self) -> str:
        return self._construct_segment(self.modules._raw)

    @staticmethod
    def _construct_segment(d: dict) -> str:

        def to_str(v) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, Sequence):
                return ', '.join(to_str(i) for i in v)
            raise ValueError("cant parse '{}'(type: {}) back to str".format(v, type(v)))

        head_format = '[{}]'
        head = None
        attr_format = '{}={}'
        body = []
        for key, val in d.items():
            if key == 'name':
                head = head_format.format(val)
            else:
                body.append(attr_format.format(key, to_str(val)))
        assert head is not None, 'cant parse a segment without head'
        return '\n'.join([head]+body)

class Conv2d(Baselayer):
    # 'conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'
    # or
    # 'conv.weight', 'conv.bias'
    def __init__(self, layer_id: int, input_layers: list, modules: nn.Module, keep_out: bool=False):
        super().__init__(layer_id, input_layers, modules, keep_out)
        self.conv_layer = self.modules.conv
        self.has_bn = False
        self.bn_scale = None
        self.is_deepwise = self.conv_layer.groups > 1
        self.divisor = 1
        self.constrain_mask = None
        self.device = self.conv_layer.state_dict()['weight'].device

        if hasattr(self.modules, 'bn'):
            self.has_bn = True
            self.bn_scale = self.modules.bn.state_dict()['weight'].abs().clone()

    def get_out_mask(self, threshold):
        if self.constrain_mask is not None:
            # 受限的卷积层
            prune_mask =  self.constrain_mask
        elif (not self.has_bn) or self.keep_out:
            # 没有bn或不剪枝的卷积层
            prune_mask = torch.ones(self.conv_layer.out_channels).bool().to(self.device)
        elif self.conv_layer.groups > 1:
            # deepwise卷积
            prune_mask = self.input_layers[0].out_mask
        else:
            # 普通卷积
            thres_index = _ceil(self.bn_scale.gt(threshold).sum().item(), self.divisor)
            # 确保剪枝之后至少有16个通道
            thres_index = max(16, thres_index)
            picked_bn_indexes = torch.sort(self.bn_scale, descending=True)[1][:thres_index]
            prune_mask = torch.zeros_like(self.bn_scale, dtype=torch.bool)
            prune_mask[picked_bn_indexes] = 1
        return prune_mask

    def prune(self, threshold) -> int:
        state_dict = list(self.modules.state_dict().values())
        input_length = len(self.input_layers)

        if input_length == 0:
            # 第一层，无输入
            input_mask = torch.ones(state_dict[0].shape[1]).bool().to(self.device)
        elif input_length == 1:
            if self.is_deepwise:
                # deepwise卷积
                assert self.conv_layer.groups == self.conv_layer.in_channels, 'only support deepwise conv'
                input_mask = torch.ones(
                    self.conv_layer.in_channels // self.conv_layer.groups
                ).bool().to(self.device)
            else:
                # 普通卷积
                input_mask = self.input_layers[0].out_mask
        else:
            raise ValueError('input of conv layer must be 0 or 1')
        self.input_mask = input_mask
        self.out_mask = self.get_out_mask(threshold)

        self.absort_channel_bias()
        self.set_channel_bias(self.out_mask)

        self.conv_layer.weight.data = state_dict[0][self.out_mask, :, :, :][:, input_mask, :, :].clone()
        if self.has_bn:
            self.clone_bn(self.modules.bn, state_dict[1:5], self.out_mask)
        else:
            self.conv_layer.bias.data = state_dict[1][self.out_mask].clone()
        if self.is_deepwise:
            self.conv_layer.groups = self.out_mask.sum().item() // input_mask.sum().item()

        # pylint: disable-msg=invalid-unary-operand-type
        return (~self.out_mask).sum().item()

    @staticmethod
    def clone_bn(bn, state_dict, mask):
        assert isinstance(bn, nn.BatchNorm2d)
        bn.weight.data = state_dict[0][mask].clone()
        bn.bias.data = state_dict[1][mask].clone()
        bn.running_mean = state_dict[2][mask].clone()
        bn.running_var = state_dict[3][mask].clone()

    def set_channel_bias(self, mask):
        if not self.has_bn:
            return
        act = nn.Identity()
        if hasattr(self.modules, 'act'):
            act = deepcopy(self.modules.act)
            if hasattr(act, 'inplace'):
                act.inplace = False
        self.const_channel_bias = ~mask * act(self.modules.bn.bias.data)

    def absort_channel_bias(self):
        if len(self.input_layers) == 0:
            return
        input_layer = self.input_layers[0]
        if input_layer.const_channel_bias is None:
            return
        sum_kernel = self.modules.conv.weight.data.sum(dim=(2, 3))
        remain_bias = input_layer.const_channel_bias.view(-1, 1)
        if self.is_deepwise:
            comp = remain_bias * sum_kernel
        else:
            comp = torch.mm(sum_kernel, remain_bias)
        comp = comp.view(-1)
        if self.has_bn:
            self.modules.bn.running_mean.sub_(comp)
        else:
            self.modules.conv.bias.data.add_(comp)

    def reflect(self) -> str:
        d = self.modules._raw
        conv_layer = self.modules.conv
        d['filters'] = conv_layer.state_dict()['weight'].shape[0]
        d['groups'] = conv_layer.groups
        return self._construct_segment(d)

class ShortCut(Baselayer):

    def prune(self, threshold) -> int:
        masks = [l.out_mask for l in self.input_layers]
        fit = all(torch.eq(x, y).all() for x, y in zip(masks, masks[1:]))
        assert fit, '{}[{}]: not all layers outmask is same'.format(self.layer_name, self.layer_id)
        self.const_channel_bias =\
            self.input_layers[0].const_channel_bias + self.input_layers[1].const_channel_bias
        self.out_mask = masks[0]
        return 0

class Route(Baselayer):

    def prune(self, threshold) -> int:
        biases = []
        for l in self.input_layers:
            if l.const_channel_bias is None:
                biases.append(torch.zeros(l.out_mask.shape[0]).to(l.out_mask.device))
            else:
                biases.append(l.const_channel_bias)
        self.const_channel_bias = torch.cat(biases, dim=0)
        self.out_mask = torch.cat([l.out_mask for l in self.input_layers])
        return 0

class Pool(Baselayer):
    pass

class Upsample(Baselayer):

    def prune(self, threshold) -> int:
        self.const_channel_bias = self.input_layers[0].const_channel_bias
        self.out_mask = self.input_layers[0].out_mask
        return 0

class YOLO(Baselayer):
    pass

class ScaleChannels(Baselayer):
    pass
