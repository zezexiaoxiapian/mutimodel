from os import path
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from dataset import augment


class BaseSampleGetter:
    '''Get sample by image path

    This is base class of a sample getter.
    One should inherit it and implement at lease `label` method.

    One can set their own train and eval dataset augment method
    by `train_augment` and `eval_augment` attribute.
    When you want to use composed augment
    (the augment need to get other sample like mixup or mosaic),
    you should set `compose_augment` attribute and overload `train` method.

    mode: in 'train', 'eval' or 'test'
    '''

    def __init__(self, mode: str='train', classes: Optional[Sequence[str]]=None):
        self.mode = mode
        self.cls_to_idx = dict(zip(classes, range(len(classes)))) if classes else None

        self.eval_augment = self.train_augment = augment.Empty()
        # not use in base class
        self.compose_augment = None

    def __call__(self, img_path: str):
        return self.caller(img_path)

    def set_mode(self, mode: str):
        self.mode = mode

    @property
    def caller(self):
        return {
            'train': self.train,
            'eval': self.eval,
            'test': self.test,
        }[self.mode]

    @property
    def is_train(self):
        return self.mode == 'train'

    @staticmethod
    def image(img_path: str):
        image = cv2.imread(img_path)
        assert image is not None, '{} not found'.format(img_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def file_name(img_path: str) -> str:
        return path.basename(img_path)

    @staticmethod
    def shape(image: np.ndarray):
        # shape in (H, W)
        return np.array(image.shape[:2], dtype=np.float32)

    def label(self, img_path: str):
        '''One should implement this method in subclass.

        It should returns bboxes (in absolutely coordinate
        [xmin, ymin, xmax, ymax, class_index] format)
        in `train` mode.
        It should returns bboxes and difficulty (0 indicates easy, 1 indicates difficult)
        in `eval` mode.

        You can judge mode by `self.is_train` in codes.
        '''
        raise NotImplementedError

    def test(self, img_path: str):
        image = self.image(img_path)
        shape = self.shape(image)
        return image, shape

    def train(self, img_path: str):
        image = self.image(img_path)
        bboxes = self.label(img_path)
        return self.train_augment(image, bboxes)

    def eval(self, img_path: str):
        image = self.image(img_path)
        shape = self.shape(image)
        image = self.eval_augment(image, [])[0]
        return (image, self.file_name(img_path), shape, *self.label(img_path))

_affine_func_T = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

def recover_bboxes_prediction(
    batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor,
    affine_func: _affine_func_T,
    onnx=False) -> torch.Tensor:
    '''
    ???????????????????????????bboxes(batch_pred_bbox)?????????????????????????????????(input_size)???
    ????????????????????????(batch_original_size)???????????????????????????bboxes?????????
    ?????????bboxes???????????????????????????????????????????????????????????????????????????????????????

    ?????????`affine_func`???????????????????????????????????????????????????bboxes?????????????????????????????????
    `affine_func`???????????????`input_size`???`batch_original_size`???
    ???????????????`delta`(()???(B, 1, 1))???`resize_ratio(B, 2)`???
    ?????????eval?????????????????????????????????????????????`affine_func`???????????????????????????????????????

    ??????????????????????????????
    - batch_pred_bbox: (B, ?, C+5)
    - input_size: (2, )
    - batch_original_size: (B, 2) or (2, )
    '''
    num_classes = batch_pred_bbox.shape[-1] - 5
    # (B, ?, 4) (B, ?, 1) (B, ?, C)
    pred_coor, pred_conf, pred_prob = batch_pred_bbox.split([4, 1, num_classes], dim=-1)

    delta, resize_ratio = affine_func(input_size, batch_original_size)
    pred_coor = pred_coor\
        .sub(delta[..., [1, 0]].repeat(1, 2).unsqueeze_(1))\
        .div_(resize_ratio) # (B, ? , 4)

    bossl = len(batch_original_size.shape)
    if bossl == 2:
        max_edge = (batch_original_size-1)[..., [1, 0]].unsqueeze_(1)
    elif bossl == 1:
        max_edge = (batch_original_size-1)[..., [1, 0]]
    if onnx:
        pred_coor = torch.cat([
            pred_coor[..., :2].clamp(0., 1e+16),
            torch.min(pred_coor[..., 2:], max_edge)],
            dim=-1,
        )
    else:
        pred_coor[..., :2].clamp_min_(0)
        pred_coor[..., 2:] = torch.min(pred_coor[..., 2:], max_edge)

    pred_prob = pred_prob.mul(pred_conf)

    bboxes = torch.cat([pred_coor, pred_prob], dim=-1) # (B, ?, 4+C)
    return bboxes

def simple_eval_augment(input_size, device):
    return augment.Compose([
        augment.Resize(input_size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(device),
    ])

def _simple_affine_bboxes(input_size: torch.Tensor, batch_original_size: torch.Tensor):
    resize_ratio, _ = (input_size / batch_original_size).min(dim=-1) # (B, )
    delta = (input_size - (resize_ratio.unsqueeze_(-1) * batch_original_size).round()) / 2 # (B, 2)
    return delta.floor(), resize_ratio.unsqueeze_(-1)

def simple_recover_bboxes_prediction(
    batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor,
    onnx=False) -> torch.Tensor:
    return recover_bboxes_prediction(
        batch_pred_bbox, input_size, batch_original_size, _simple_affine_bboxes, onnx,
    )