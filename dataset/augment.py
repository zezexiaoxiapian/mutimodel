from collections import namedtuple
from itertools import chain
from math import ceil
from typing import Any, Callable, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from numpy import random
import math
import tools

_size_T = Union[List[int], Tuple[int, int]]
_triplet_T = Union[List[float], Tuple[float, float, float]]
_range_T = Union[List[float], Tuple[float, float]]
_transform_T = Callable[[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]
_sampler_T = Callable[[], Tuple[np.ndarray, np.ndarray]]
_bboxes_T = Union[List, np.ndarray]
_aware_size_T = Union[_size_T, Callable[[], _size_T]]
_ratio_T = Union[float, List[float], Tuple[float, float]]


def _filter_bboxes_by_iou_area_ratio(original_bboxes: _bboxes_T, new_bboxes: _bboxes_T,
    iou_threshold=0.3, area_threshold=56, ratio_threshold=10) -> np.ndarray:
    w = new_bboxes[:, 2] - new_bboxes[:, 0]
    h = new_bboxes[:, 3] - new_bboxes[:, 1]
    area = w * h
    area0 = (original_bboxes[:, 2] - original_bboxes[:, 0]) *\
        (original_bboxes[:, 3] - original_bboxes[:, 1])
    ratio = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
    i = (area > area_threshold) & (area / (area0 + 1e-16) > iou_threshold) & (ratio < ratio_threshold)
    return new_bboxes[i]

def _resolve_aware_size(aware_size: _aware_size_T) -> _size_T:
    return aware_size() if callable(aware_size) else aware_size

def _resolve_ratio(ratio: _ratio_T) -> _range_T:
    try:
        iter(ratio)
    except TypeError:
        return (ratio, ratio)
    return ratio

def quantize_number(n, q: int, round_func=round) -> int:
    """Converts a number to closest non-zero int divisible by q."""
    return int(round_func(n / q) * q)

class RandomCrop:

    def __init__(self, size: _size_T, p=0.5,
        iou_threshold=0.3, area_threshold=56, ratio_threshold=10):
        self.size = size
        self.p = p
        self.iou_threshold = iou_threshold
        self.area_threshold = area_threshold
        self.ratio_threshold = ratio_threshold

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h, w = img.shape[:2]
        ch, cw = self.size
        crop_xmin = random.randint(0, max(w - cw, 0) + 1)
        crop_ymin = random.randint(0, max(h - ch, 0) + 1)
        crop_xmax = min(crop_xmin + cw, w)
        crop_ymax = min(crop_ymin + ch, h)
        img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

        if len(bboxes) == 0:
            return img, bboxes

        new_bboxes = bboxes.copy()
        new_bboxes[:, [0, 2]] = np.clip(new_bboxes[:, [0, 2]] - crop_xmin, 0, cw)
        new_bboxes[:, [1, 3]] = np.clip(new_bboxes[:, [1, 3]] - crop_ymin, 0, ch)
        new_bboxes = _filter_bboxes_by_iou_area_ratio(
            bboxes, new_bboxes, iou_threshold=self.iou_threshold,
            area_threshold=self.area_threshold, ratio_threshold=self.ratio_threshold
        )

        return img, new_bboxes

class RandomSafeCrop:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h, w = img.shape[:2]
        # 得到可以包含所有bbox的最小bbox
        if len(bboxes) > 0:
            max_bbox = np.concatenate([
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_bbox = np.round(max_bbox)
        else:
            cx, cy = w // 2, h // 2
            max_bbox = np.array([cx, cy, cx+1, cy+1])
        crop_xmin = random.randint(0, max(0, max_bbox[0])+1)
        crop_ymin = random.randint(0, max(0, max_bbox[1])+1)
        crop_xmax = random.randint(min(w, max_bbox[2]), w+1)
        crop_ymax = random.randint(min(h, max_bbox[3]), h+1)

        img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes

class RandomHFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        w = img.shape[1]
        img = img[:, ::-1, :]

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return img, bboxes

class RandomVFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h = img.shape[0]
        img = img[::-1, :, :]

        if len(bboxes) != 0:
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        return img, bboxes

class ColorJitter:

    def __init__(self, hue: float=0.5, saturation: float=0.5, value: float=0.5, p=1.):
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        the_img = img[..., 3:]
        img = img[..., :3]
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.saturation, self.value] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(np.uint8)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return np.concatenate([img, the_img], axis=-1), bboxes

class CutOut:

    def __init__(self, size: int, n_holes: int, p: float=0.5, pad_val: int=128):
        self.p = p
        self.size = size // 2
        self.n_holes = n_holes
        self.pad_val = pad_val

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h, w = img.shape[:2]
        for _ in range(self.n_holes):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)

            y1 = np.clip(y - self.size, 0, h)
            y2 = np.clip(y + self.size, 0, h)
            x1 = np.clip(x - self.size, 0, w)
            x2 = np.clip(x + self.size, 0, w)

            img[y1:y2, x1:x2, :] = self.pad_val
        return img, bboxes

class Normalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = img.astype(np.float32, copy=False)
        img = (img/255. - self.mean) / self.std
        return img, bboxes

class DeNormalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        np.clip((img * self.std + self.mean)*255., 0, 255, out=img)
        return img.astype(np.uint8), bboxes

class Resize:

    def __init__(self, size: _aware_size_T, pad_val: int=128, nopad: bool=False):
        self.pad_val = pad_val
        self.size = size
        self.nopad = nopad

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        target_h, target_w = _resolve_aware_size(self.size)
        img_h, img_w = img.shape[:2]

        resize_ratio = min(target_w / img_w, target_h / img_h)
        resize_w = round(resize_ratio * img_w)
        resize_h = round(resize_ratio * img_h)
        image_resized = cv2.resize(img, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        if self.nopad:
            dl = du = 0
            img_padded = image_resized
        else:
            dl = (target_w - resize_w) // 2
            dr = target_w - resize_w - dl
            du = (target_h - resize_h) // 2
            dd = target_h - resize_h - du
            img_padded = np.pad(
                image_resized, ((du, dd), (dl, dr), (0, 0)),
                'constant', constant_values=self.pad_val
            )

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dl
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + du
        return img_padded, bboxes

class ResizeRatio:

    def __init__(self, size: _ratio_T):
        self.ratio = _resolve_ratio(size)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        target_h, target_w = list(map(lambda x: round(x[0] * x[1]), zip(self.ratio, img.shape[:2])))
        image_resized = cv2.resize(img, dsize=(target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] *= self.ratio[1]
            bboxes[:, [1, 3]] *= self.ratio[0]
        return image_resized, bboxes

class PadNearestDivisor:

    def __init__(self, pad_val: int=128, divisor: int=32):
        self.pad_val = pad_val
        self.divisor = divisor

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img_h, img_w = img.shape[:2]
        target_h = quantize_number(img_h, self.divisor, round_func=ceil)
        target_w = quantize_number(img_w, self.divisor, round_func=ceil)

        dl = (target_w - img_w) // 2
        dr = target_w - img_w - dl
        du = (target_h - img_h) // 2
        dd = target_h - img_h - du
        img_padded = np.pad(
            img, ((du, dd), (dl, dr), (0, 0)),
            'constant', constant_values=self.pad_val
        )

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] += dl
            bboxes[:, [1, 3]] += du
        return img_padded, bboxes

class Mixup:

    def __init__(self, sampler: _sampler_T, p=0.5, beta: float=1.):
        self.sampler = sampler
        self.p = p
        self.beta = beta

    @staticmethod
    def mixup_bboxes(bboxes: _bboxes_T, mixup_factor: float):
        if len(bboxes) == 0:
            return bboxes
        mfs = np.full((len(bboxes), 1), mixup_factor, dtype=np.float32)
        return np.concatenate([bboxes, mfs], axis=-1)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            bboxes = self.mixup_bboxes(bboxes, 1.0)
            return img, bboxes
        img_mix, bboxes_mix = self.sampler()
        lam = random.beta(self.beta, self.beta)
        img = lam * img + (1 - lam) * img_mix
        bboxes = self.mixup_bboxes(bboxes, lam)
        bboxes_mix = self.mixup_bboxes(bboxes_mix, 1-lam)
        bboxes_no_empty = [b for b in [bboxes, bboxes_mix] if len(b) != 0]
        if len(bboxes_no_empty) == 0:
            return img, np.zeros([1, 6], dtype=np.float32)
        bboxes = np.concatenate(bboxes_no_empty)
        # tools.draw_bboxes_on_image(img, bboxes, 'results/mosaic_mixup.jpg')
        # assert 0
        return img.astype(np.float32), bboxes

class Mosaic:

    def __init__(self, sampler: _sampler_T, size: _aware_size_T, pad_val: int=128, p: float=1):
        self.sampler = sampler
        self.size = size
        self.pad_val = pad_val
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        input_h, input_w = _resolve_aware_size(self.size)
        xc = int(random.uniform(input_w * 0.5, input_w * 1.5))
        yc = int(random.uniform(input_h * 0.5, input_h * 1.5))

        bboxes4 = []
        img4 = np.full((input_h * 2, input_w * 2, 4), self.pad_val, dtype=np.uint8)
        other_imgs, other_bboxes = list(zip(*[self.sampler() for _ in range(3)]))
        all_imgs_bboxes = zip(chain([img], other_imgs), chain([bboxes], other_bboxes))
        for i, (image, bboxes) in enumerate(all_imgs_bboxes):
            h, w = image.shape[:2]

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            bboxes[:, [0, 2]] += x1a - x1b
            bboxes[:, [1, 3]] += y1a - y1b
            bboxes4.append(bboxes)

        bboxes4 = np.concatenate(bboxes4, axis=0)
        bboxes4[:, [0, 2]] = np.clip(bboxes4[:, [0, 2]], 0, 2 * input_w)
        bboxes4[:, [1, 3]] = np.clip(bboxes4[:, [1, 3]], 0, 2 * input_h)

        # tools.draw_bboxes_on_image(img4, bboxes4, 'results/mosaic.jpg')
        # assert 0

        return img4, bboxes4

class RandomAffine:

    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, border_ratio=0.25, pad_val: int=128):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border_ratio = border_ratio
        self.pad_val = pad_val

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img_h, img_w, _ = img.shape
        border = (self.border_ratio * img_h, self.border_ratio * img_w)
        out_h, out_w = int(img_h - 2 * border[0]), int(img_w - 2 * border[1])

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img_h / 2, img_w / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-self.translate, self.translate) * img_w - border[1]  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * img_h - border[0]  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        img = cv2.warpAffine(
            img, M[:2], dsize=(out_w, out_h), flags=cv2.INTER_LINEAR,
            borderValue=(self.pad_val, self.pad_val, self.pad_val, self.pad_val),
        )

        n = len(bboxes)
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, out_w)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, out_h)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        bboxes = bboxes[i]
        bboxes[:, :4] = xy[i]
        return img, bboxes

class GenRadar:

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        h, w, _ = img.shape
        radar = np.zeros((h, w), dtype=np.float32)
        for box in bboxes:
            x1, y1, x2, y2 = box[:4]
            xc, yc = int((x1+x2)/2), int((y1+y2)/2)
            ww, hh = int(x2-x1), int(y2-y1)
            xaxis, yaxis = np.meshgrid(np.arange(w), np.arange(h))
            m = np.stack([yaxis, xaxis], axis=-1)
            radar += np.exp(-0.5*(np.square((m[..., 1]-xc)/ww*3) + np.square((m[..., 0]-yc)/hh*3)))
            # radar[yc-hh:yc+hh, xc-wh:xc+wh] = 0.3 + random.rand() * 0.7
        radar += random.rand(h, w) * 0.5
        img = np.concatenate([img, (np.clip(radar, 0, 1)*255).astype(np.uint8)[..., None]], axis=-1)
        return img, bboxes

class ToTensor:

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
        return img, bboxes

class HWCtoCHW:

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img, bboxes

class Empty:

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        return img, bboxes

class Compose:

    def __init__(self, transforms: Sequence[_transform_T]):
        self.transforms = transforms

    def __call__(self, img, bboxes: _bboxes_T):
        for transform in self.transforms:
            img, bboxes = transform(img, bboxes)
        return img, bboxes
