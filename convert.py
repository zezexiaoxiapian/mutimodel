import numpy as np
import onnx
import torch
from torch import nn

import tools
from export.onnx_exporter import ONNXExporter
from dataset import RECOVER_BBOXES_REGISTER


def save_weight_to_darknet(weight_path: str, save_path: str, seen: int=0):
    fw = open(save_path, 'wb')

    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']

    header = np.array([0, 0, 0, seen], dtype=np.int32)
    header.tofile(fw)

    pre_conv = None
    pre_bias = []
    for key, params in state_dict.items():
        params_shape = len(params.shape)
        if params_shape == 4: # conv weight
            if pre_conv is not None:
                pre_conv.numpy().tofile(fw)
            pre_conv = params
        elif params_shape == 1: #BN or conv_bias
            if key.endswith('bias') and len(pre_bias) == 0: # conv_bias
                params.numpy().tofile(fw)
                pre_conv.numpy().tofile(fw)
                pre_conv = None
            else: # BN
                pre_bias.append(params)
                if len(pre_bias) == 4:
                    pre_bias[1].numpy().tofile(fw)
                    pre_bias[0].numpy().tofile(fw)
                    pre_bias[2].numpy().tofile(fw)
                    pre_bias[3].numpy().tofile(fw)
                    pre_bias.clear()
                    assert pre_conv is not None
                    pre_conv.numpy().tofile(fw)
                    pre_conv = None
        else:
            pass

    if pre_conv is not None:
        pre_conv.numpy().tofile(fw)

    fw.close()

def export_qat_to_quantized(cfg_path: str, weight_path: str, save_path: str):
    model, model_info = tools.build_model(
        cfg_path, weight_path, device='cpu', dataparallel=False, quantized=True, backend='qnnpack'
    )
    model_info['model'] = model.state_dict()
    model_info['type'] = 'quant'
    torch.save(model_info, save_path)

def export_quantized_to_onnx(cfg_path: str, weight_path: str, onnx_path: str):
    model = tools.build_model(
        cfg_path, weight_path, device='cpu', dataparallel=False, quantized=True, backend='qnnpack'
    )[0]
    model.eval()
    onnx_exporter = ONNXExporter(model, size=(512, 512))
    onnx_model = onnx_exporter.export(graph_name='quantized-mobilenetv2-yolov3-lite')
    onnx.save(onnx_model, onnx_path)

def export_normal_to_onnx(cfg_path: str, weight_path: str, onnx_path: str):
    img_size = (512, 512)
    ori_size = (512, 512)

    model = tools.build_model(
        cfg_path, weight_path, device='cpu', dataparallel=False, onnx=True,
    )[0]
    model.eval()

    def fusion(mod):
        if getattr(mod, '_type', None) == 'convolutional' and hasattr(mod, 'bn'):
            mod.conv = nn.utils.fuse_conv_bn_eval(mod.conv, mod.bn)
            mod.bn = nn.Identity()

    for m in model.module_list.children():
        fusion(m)

    def preprocess(img):
        target_h, target_w = img_size
        img_h, img_w = img.shape[2:]
        resize_ratio = torch.min(target_w / img_w.float(), target_h / img_h.float())
        resize_w = (resize_ratio * img_w).long()
        resize_h = (resize_ratio * img_h).long()
        image_resized = torch.functional.F.interpolate(img, (resize_w, resize_h))

        dl = (target_w - resize_w) // 2
        dr = target_w - resize_w - dl
        du = (target_h - resize_h) // 2
        dd = target_h - resize_h - du
        img_padded = torch.zeros((1, 3, target_h, target_w), dtype=torch.float32)
        img_padded[..., du:du+resize_h, dl:dl+resize_w] = image_resized
        return img_padded

    class InferModel(nn.Module):

        def __init__(self, m, img_size):
            super().__init__()
            self.m = m
            self.size = torch.tensor(img_size, dtype=torch.float32)
            self.ori_size = torch.tensor(ori_size, dtype=torch.float32)
            self.preprocess_fn = preprocess
            self.recover_fn = RECOVER_BBOXES_REGISTER['voc']

        def forward(self, x):
            batch_bboxes = self.m(x)
            # (B, ?, 4) (B, ?, 1) (B, ?, C)
            pred_coor, pred_conf, pred_prob = batch_bboxes[..., :4], batch_bboxes[..., 4:5], batch_bboxes[..., 5:]
            pred_prob.mul_(pred_conf)
            mv, mi = torch.max(pred_prob, dim=-1, keepdim=True)
            return torch.cat([pred_coor, mi.float(), mv], dim=-1)
    imodel = InferModel(model, img_size)
    imodel.eval()

    torch_in = torch.randn(1, 3, *ori_size)
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(
        imodel, torch_in, onnx_path, verbose=False, input_names=['input'],
        output_names=['output'], dynamic_axes=dynamic_axes, opset_version=11,
    )

def partial(weight_path: str, save_path: str, layers: int):
    state_dict = torch.load(weight_path, 'cpu')['model']
    partial_dict = {}
    sentinel = f'{layers+1}.'
    for key, params in state_dict.items():
        if sentinel in key:
            break
        partial_dict[key] = params
    print('last layer name: {}({})'.format(key, list(params.shape)))
    torch.save(partial_dict, save_path)

def make_backbone(cfg_path: str, weight_path: str, save_path: str):
    state_dict = torch.load(weight_path, 'cpu')['model_state']
    model = tools.build_model(cfg_path, device='cpu', dataparallel=False)[0]
    new_state_dict = {}
    for (bn, bp), (mn, mp) in zip(state_dict.items(), model.state_dict().items()):
        if not bp.shape == mp.shape:
            print(f'last layer: {bn}({list(bp.shape)}) -> {mn}({list(mp.shape)})')
            break
        new_state_dict[mn] = bp
    torch.save(new_state_dict, save_path)

if __name__ == "__main__":
    weight_path = '/home/tjz/PycharmProjects/Det (2)/weights/model-89-0.3512.pt'
    # partial(weight_path, 'weights/pretrained/mobilev2-uav-prune75.pt', 62)
    # export_qat_to_quantized(
    #     'model/cfg/mobilenetv2-fpn-uav.cfg',
    #     'weights/uav_mobilenetv2_fpn_l1_qat/model-59-0.7609.pt',
    #     'weights/uav_mobilenetv2_fpn_l1_qat/model-59-0.7609-quant.pt'
    # )
    # weight_path = 'weights/VOC_quant3/model-44.pt'
    # weight_path = 'weights/trained/model-74-0.7724.pt'
    # save_weight_to_darknet(weight_path, weight_path.rsplit('.', 1)[0]+'-convert.weights')
    # export_quantized_to_onnx('model/cfg/myolo-prune-40.cfg', weight_path, 'export/quant_myolov1.onnx')
    export_normal_to_onnx('/home/tjz/PycharmProjects/Det (2)/model/cfg/pcspnet-spp-pan-flir.cfg', weight_path, 'export/tjz.onnx')
    # make_backbone('/home/eleflea/code/pycls/weights/regnety_400m/checkpoints/model_epoch_0100.pyth', 'model/cfg/regnety-400m-fpn.cfg', 'weights/pretrained/regnety_400m.pt')
