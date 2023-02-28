
import heapq
import math
import os
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from copy import deepcopy
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, TypeVar
from model.interpreter import DetectionModel
import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import ops
from tqdm import tqdm

# dont directly import interpreter and DetectionModel cause circular import
#from model.interpreter import DetectionModel

try:
    from time import time_ns
except ImportError:
    from time import time

    def time_ns():
        return int(time() * 1e9)


class ModelEMA:


    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = bare_model(model).state_dict()  # model state_dict
            for k, v in bare_model(self.ema).state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

class BBoxDrawer:
    '''Draw bboxes on image
    '''
    def __init__(self, border_thickness=2, font=cv2.FONT_HERSHEY_PLAIN, font_color=(255, 255, 255),
        text_scale=1.0, text_thickness=1, alpha=0.75, palette=None):
        self.border_thickness = border_thickness
        self.font = font
        self.font_color = font_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.alpha = alpha
        self.palette = np.random.RandomState(123).randint(0, 256, (256, 3), dtype=np.uint8) if palette is None else palette

    def __call__(self, image, text, bbox, color):
        '''*INPLACE* method
        '''
        assert image.dtype == np.uint8, 'except a numpy uint8 array, but {}'.format(image.dtype)
        img_h, img_w, *_ = image.shape
        x1, y1, x2, y2 = [int(c) for c in bbox]
        if x1 >= img_w or y1 >= img_h:
            return image
        bt = self.border_thickness
        if isinstance(color, int):
            color = tuple(int(v) for v in self.palette[color])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, bt)
        if not text:
            return image
        text_loc = (max(int(x1+bt+0.5), 0), max(int(y1+bt+0.5), 0))
        margin = 3
        size = cv2.getTextSize(text, self.font, self.text_scale, self.text_thickness)
        w = size[0][0] + margin * 2
        h = size[0][1] + margin * 2
        # the patch is used to draw boxed text
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        patch[...] = color
        cv2.putText(patch, text, (margin+1, h-margin-2), self.font, self.text_scale,
                    self.font_color, thickness=self.text_thickness, lineType=cv2.LINE_8)
        w = min(w, img_w - text_loc[0])  # clip overlay at image boundary
        h = min(h, img_h - text_loc[1])
        # Overlay the boxed text onto region of interest (roi) in img
        roi = image[text_loc[1]:text_loc[1]+h, text_loc[0]:text_loc[0]+w, :]
        cv2.addWeighted(patch[0:h, 0:w, :], self.alpha, roi, 1 - self.alpha, 0, roi)
        return image

def draw_bboxes_on_image(image, bboxes: np.ndarray, save_path: str):
    # TODO: deprecation, merge it to BBoxDrawer
    for box in bboxes:
        x1, y1, x2, y2, cls_index = [int(n) for n in box][:5]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = str(cls_index)
        cv2.putText(image, text, (x1, y1-5), 0, 0.4, (0, 255, 0))
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)

def add_losses_to_summary_writer(writer: SummaryWriter, losses: dict, global_step):
    tag_dict = {
        'main': losses['loss'].mean(),
        'bbox': losses['xy_loss'].mean(),
        'object': losses['obj_loss'].mean(),
        'class': losses['cls_loss'].mean(),
    }
    tag_dict.update({'branch/{}'.format(i): branch_loss.mean()
        for i, branch_loss in enumerate(losses['loss_per_branch'], 1)})
    writer.add_scalars('train/loss', tag_dict, global_step=global_step)

AP = namedtuple('AP', ['mAPs', 'APs', 'AP', 'raw', 'class_names', 'iou_thresholds'])

def add_AP_to_summary_writer(writer: SummaryWriter, ap: AP, global_step):
    tag_dict = {'AP': ap.AP}
    tag_dict.update({'class_AP/{}'.format(cls_name): cls_ap
        for cls_name, cls_ap in zip(ap.class_names, ap.APs)})
    tag_dict.update({'iou_AP/{:.2f}'.format(iou): iou_ap
        for iou, iou_ap in zip(ap.iou_thresholds, ap.mAPs)})
    writer.add_scalars('eval/AP', tag_dict, global_step=global_step)

def print_metric(metric: AP, verbose=True):
    def pad_element(x, w: int):
        sx = str(x)
        return sx + ' ' * (w - len(sx))

    def format_percents(fs):
        return ['{:.2f}'.format(f * 100) for f in fs]

    iou_thres = metric.iou_thresholds
    raw = metric.raw
    if verbose:
        class_names = metric.class_names
        capitial = 'CLASS\\IOU'
        col1_width = max(len(capitial), max(len(n) for n in class_names)) + 2
    else:
        class_names = []
        capitial = 'IOU'
        col1_width = 6
    cols_width = [col1_width] + [7 for _ in range(len(iou_thres))] + [5]
    rows = [[capitial] + np.round(iou_thres * 100).astype(np.int).tolist() + ['APs']]
    for i, name in enumerate(class_names):
        r = [name] + format_percents(raw[i].tolist() + [metric.APs[i]])
        rows.append(r)
    rows.append(['mAPs'] + format_percents(metric.mAPs.tolist() + [metric.AP]))
    for r in rows:
        print(''.join(pad_element(e, w) for w, e in zip(cols_width, r)))

def is_sequence(x: Any) -> bool:
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True

def compute_time(model: nn.Module, times: int=64, input_size=(3, 512, 512), batch_size: int=1, device='cuda'):
    torch.backends.cudnn.benchmark = True
    model.eval()
    input_shape = [batch_size] + list(input_size)
    inputs = torch.zeros(*input_shape).to(device)

    timer = TicToc()
    warmup_iter = 10
    with torch.no_grad():
        for cur_iter in tqdm(range(times + warmup_iter)):
            # Reset the timers after the warmup phase
            if cur_iter == warmup_iter:
                timer.reset()
            # Forward
            timer.tic()
            model(inputs)
            torch.cuda.synchronize()
            timer.toc()
    torch.backends.cudnn.benchmark = False
    # ns -> ms
    return timer.mean() / 1e6

def _state_dict_is_dp(state_dict: Dict) -> bool:

    for k in state_dict:
        return k.startswith('module.')

def _model_is_dp(model: nn.Module) -> bool:
    return isinstance(model, nn.DataParallel)

def load_weight(model: nn.Module, state_dict: Dict):
    dp_model = _model_is_dp(model)
    dp_state_dict = _state_dict_is_dp(state_dict)
    if dp_model == dp_state_dict:
        model.load_state_dict(state_dict)
        return
    if dp_model:
        model.module.load_state_dict(state_dict)
    else:
        nn.DataParallel(model).load_state_dict(state_dict)

def load_backbone(model: nn.Module, state_dict: Dict):
    dp_model = _model_is_dp(model)
    dp_state_dict = _state_dict_is_dp(state_dict)
    if dp_model == dp_state_dict:
        ori_state_dict = model.state_dict()
        ori_state_dict.update(state_dict)
        model.load_state_dict(ori_state_dict)
        return
    if dp_model:
        load_backbone(model.module, state_dict)
    else:
        load_backbone(nn.DataParallel(model), state_dict)

def build_model(cfg_path: Optional[str], weight_path: Optional[str]=None, backbone_path: Optional[str]=None,
    clear_history: bool=False, device='cuda', dataparallel: bool=True, device_ids=None,
    qat: bool=False, backend: Optional[str]=None, quantized: bool=False, onnx: bool=False) -> (nn.Module, Dict):

    # state_dict中除权重的其他信息，默认为空
    model_info = {}

    if weight_path:
        state_dict = torch.load(weight_path, map_location=device)
        state_dict_type = state_dict.get('type', 'normal')
        weight = state_dict['model']
        # 把其他信息找出来
        model_info = {k: v for k, v in state_dict.items() if k != 'model'}
        if clear_history:
            # 清除step
            model_info['step'] = 0
    else:
        state_dict_type = None
    if cfg_path:
        cfg = cfg_path
    else:
        # 如果cfg_path没有的话，从state_dict中的'cfg'参数加载cfg
        cfg = StringIO(state_dict['cfg'])

    # 权重是QAT或量化的，或者要返回QAT或量化模型，都需要做fuse和prepare_qat
    is_need_fuse = state_dict_type in {'qat', 'quant'} or qat or quantized
    # qat或量化模型都需要替换模型的一些部分，比如ReLU6->ReLU
    model = DetectionModel(cfg, is_need_fuse, onnx)
    if dataparallel:
        model = torch.nn.DataParallel(model, device_ids)

    if backbone_path:
        print('loading backbone weights from {}'.format(backbone_path))
        backbone_state_dict = torch.load(backbone_path, map_location=device)
        load_backbone(model, backbone_state_dict)

    if state_dict_type == 'normal':
        load_weight(model, weight)
        print('resumed at %d steps from %s' % (model_info['step'], weight_path))

    if is_need_fuse:
        fuse_model(model, inplace=True)
        if backend is None:
            backend = model_info.get('backend')
            if backend not in {'fbgemm', 'qnnpack'}:
                backend = 'qnnpack'
        prepare_qat(model, backend=backend, inplace=True)
    if state_dict_type == 'qat':
        load_weight(model, weight)
        print('resumed qat at %d steps from %s' % (model_info['step'], weight_path))

    if state_dict_type == 'quant' or quantized:
        quantized_model(model, inplace=True)
    if state_dict_type == 'quant':
        load_weight(model, weight)

    return model.to(device), model_info

def _condition_copy_model(model: nn.Module, inplace: bool=True) -> nn.Module:
    if inplace:
        return model
    new_model = deepcopy(model)
    return new_model

def bare_model(model: nn.Module):
    if _model_is_dp(model):
        return model.module
    return model

def fuse_model(model: nn.Module, inplace: bool=True) -> nn.Module:
    new_model = _condition_copy_model(model, inplace)
    bared_model = bare_model(new_model)
    for layer in bared_model.module_list:
        if layer._type == 'convolutional':
            names = [name for name, _ in layer.named_children() if name in {'conv', 'bn', 'act'}]
            if len(names) < 2:
                continue
            torch.quantization.fuse_modules(layer, names, inplace=True)
    return new_model

def prepare_qat(model: nn.Module, backend: str='fbgemm', inplace: bool=True) -> nn.Module:
    new_model = _condition_copy_model(model, inplace)
    new_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    return torch.quantization.prepare_qat(new_model, inplace=inplace)

def quantized_model(model: nn.Module, inplace: bool=False) -> nn.Module:
    new_model = _condition_copy_model(model, inplace)
    # 目前pytorch量化模型只支持CPU推理
    quant_model = torch.quantization.convert(bare_model(new_model).eval().cpu(), inplace=inplace)
    return quant_model.eval()

def print_quantized_state_dict(state_dict: Dict[str, Any]):
    for k, v in state_dict.items():
        if k.split('.')[-1] in {'scale', 'zero_point'}:
            print(f'{k}<{v.dtype}>: {v.detach().numpy()}')
        elif k.split('.')[-1] == 'weight':
            print(f'{k}<{v.dtype}>: {list(v.shape)}'
                f'(scale={v.q_scale()}, zero_point={v.q_zero_point()})')
        else:
            print(f'{k}<{v.dtype}>: {list(v.shape)}')

def get_bn_layers(model: nn.Module) -> List[nn.BatchNorm2d]:
    bn_layers = []
    c = 0
    for l in bare_model(model).module_list:
        if l._type == 'convolutional' and hasattr(l, 'bn'):
            if not hasattr(l, '_notprune'):
                bn_layers.append(l.bn)
            c += 1
    print("sparse mode: {}/{} BN layers will be sparsed.".format(len(bn_layers), c))
    return bn_layers

def iou_calc1(boxes1: np.ndarray, boxes2: np.ndarray):
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / np.maximum(union_area, 1e-14)
    return IOU

def iou_calc3(boxes1: torch.Tensor, boxes2: torch.Tensor):
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area
    return IOU

def giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / (union_area + 1e-16)

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - (enclose_area - union_area) / (enclose_area + 1e-16)

    return GIOU

def diou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - (enclose_area - union_area) / enclose_area

    center_b1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    center_b2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    distance_center = (center_b1 - center_b2).pow(2).sum(dim=-1)
    distance_enclose = (enclose_left_up - enclose_right_down).pow(2).sum(dim=-1)

    return GIOU + distance_center / distance_enclose

def ciou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    b1_w, b1_h = boxes1[..., 2] - boxes1[..., 0], boxes1[..., 3] - boxes1[..., 1]
    b2_w, b2_h = boxes2[..., 2] - boxes2[..., 0], boxes2[..., 3] - boxes2[..., 1]
    boxes1_area = b1_w * b1_h
    boxes2_area = b2_w * b2_h

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - (enclose_area - union_area) / enclose_area

    center_b1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    center_b2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    distance_center = (center_b1 - center_b2).pow(2).sum(dim=-1)
    distance_enclose = (enclose_left_up - enclose_right_down).pow(2).sum(dim=-1)

    v = (4 / (math.pi ** 2)) * (torch.atan(b1_w / b1_h) - torch.atan(b2_w / b2_h)).pow(2)
    with torch.no_grad():
        s = 1 - IOU
        alpha = v / (s + v)

    return GIOU + distance_center / distance_enclose + alpha * v

def iou_xywh_numpy(boxes1: np.ndarray, boxes2: np.ndarray):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，
    # 所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area
    return IOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)

def torch_nms(bboxes: torch.Tensor, score_threshold: float,
    iou_threshold: float) -> torch.Tensor:
    class_scores = bboxes[:, 4:]
    mask = class_scores > score_threshold
    indexes = mask.nonzero(as_tuple=False)
    pick_scores = class_scores[mask]
    pick_class_indexes = indexes[:, 1]
    pick_bboxes = bboxes[:, :4][indexes[:, 0]]
    keep = ops.boxes.batched_nms(
        pick_bboxes, pick_scores, pick_class_indexes, iou_threshold
    )
    if keep.numel() == 0:
        # pylint: disable-msg=not-callable
        return torch.tensor([]).to(bboxes)
    return torch.cat([
        pick_bboxes[keep],
        pick_scores[keep, None],
        pick_class_indexes[keep, None].float(),
    ], dim=1)

class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, temp_sum, n=1):
        self.sum += temp_sum
        self.count += n

    def get_avg_reset(self):
        if self.count == 0:
            return 0.
        avg = float(self.sum) / float(self.count)
        self.reset()
        return avg

    def get_sum_reset(self):
        s = self.sum
        self.sum = 0
        return s

class TicToc:

    def __init__(self, name: Optional[str]=None):
        self.name = name
        self.last = 0
        self.records = []
        self.reset()

    def reset(self):
        self.last = 0
        self.records.clear()

    def tic(self):
        self.last = time_ns()

    def toc(self):
        self.records.append(time_ns() - self.last)

    def __getitem__(self, index):
        return self.records[index]

    def mean(self):
        return np.mean(self.records)

    def mean_reset(self):
        m = self.mean()
        self.reset()
        return m

    def sum(self):
        return np.sum(self.records)

    def sum_reset(self):
        s = self.sum()
        self.reset()
        return s

    def statistics(self):
        std = np.std(self.records)
        return {
            'name': 'none' if self.name is None else self.name,
            'mean': np.mean(self.records),
            'std': std,
            '3std': 3*std,
            'min': np.amin(self.records),
            'max': np.amax(self.records),
        }

class _Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

    @abstractmethod
    def __neg__(self): ...

_comparable_T = TypeVar('Comparable_T', bound=_Comparable)
_key_func_T = Optional[Callable[[Any], _comparable_T]]
_val_T = TypeVar('val_T')

class PriorityQueue:

    def __init__(self, key: _key_func_T):
        self.key = (lambda x: x) if key is None else key
        self.queue = []
        self._index = 0

    def push(self, val: _val_T):
        priority = self.key(val)
        heapq.heappush(self.queue, (priority, self._index, val))
        self._index += 1

    def pop(self) -> _val_T:
        return heapq.heappop(self.queue)[-1]

    def __len__(self):
        return len(self.queue)

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration

    def __iter__(self):
        return self

def ensure_dir(path: str):
    # 建立文件夹（如果不存在的话）
    os.makedirs(path, exist_ok=True)
