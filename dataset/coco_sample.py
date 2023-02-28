from typing import Callable
import numpy as np
from yacs.config import CfgNode as CN

from dataset import augment
from dataset.base_sample import (BaseSampleGetter, simple_eval_augment,
                                 simple_recover_bboxes_prediction)


class COCOSampleGetter(BaseSampleGetter):
    '''COCO dataset
    get sample by image path

    use darknet labels (class, xc, yc, w, h) in relative

    mode: in 'train', 'eval' or 'test'
    '''

    def label(self, img_path: str):
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        bbs, diffs = [], []
        fr = open(label_path, 'r')
        for line in fr.readlines():
            ann = line.split(' ')
            # diff always be 0
            diff = 0
            cls_idx = int(ann[0])
            # note here we return a relative bboxes
            # we turn it to absolutely in `self._train` and `self.eval`
            half_rw, half_rh = float(ann[3]) / 2, float(ann[4]) / 2
            rx1 = float(ann[1]) - half_rw
            ry1 = float(ann[2]) - half_rh
            rx2 = float(ann[1]) + half_rw
            ry2 = float(ann[2]) + half_rh
            box = [rx1, ry1, rx2, ry2, cls_idx]
            bbs.append(box)
            diffs.append(diff)
        fr.close()
        bbs = np.array(bbs, dtype=np.float32)
        if self.is_train:
            return bbs
        return bbs, np.array(diffs)

    @staticmethod
    def _relative_to_absolute(bboxes: np.ndarray, shape: np.ndarray):
        bboxes[:, :-1] *= np.tile(shape[[1, 0]], 2)
        return bboxes

    def set_train_augment(self, augment_cfg: CN, input_size, img_path_sampler):
        self.train_augment = augment.Resize(input_size, nopad=True)
        mosaic_sampler = lambda: self._train(img_path_sampler())
        mixup_augment = augment.Compose([
            augment.Mosaic(mosaic_sampler, p=1, size=input_size),
            augment.RandomAffine(
                degrees=augment_cfg.degrees,
                translate=augment_cfg.translate,
                scale=augment_cfg.scale,
                shear=augment_cfg.shear,
            ),
            augment.ColorJitter(
                hue=augment_cfg.hue,
                saturation=augment_cfg.saturation,
                value=augment_cfg.value,
            ),
            augment.RandomHFlip(p=augment_cfg.hflip_p),
            augment.RandomVFlip(p=augment_cfg.vflip_p),
        ])
        mixup_sampler = lambda: self.sample_with_aug(img_path_sampler(), mixup_augment)
        self.compose_augment = augment.Compose([
            mixup_augment,
            augment.Mixup(mixup_sampler, p=augment_cfg.mixup_p, beta=1.5),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            augment.ToTensor('cpu'),
        ])
        return self

    def set_eval_augment(self, input_size):
        self.eval_augment = eval_augment_coco(input_size, 'cpu')
        return self

    def _train(self, img_path: str):
        image = self.image(img_path)
        bboxes = self._relative_to_absolute(self.label(img_path), self.shape(image))
        return self.train_augment(image, bboxes)

    def train(self, img_path: str):
        image, bboxes = self._train(img_path)
        return self.compose_augment(image, bboxes)

    def sample_with_aug(self, img_path: str, augment: Callable):
        image = self.image(img_path)
        bboxes = self.label(img_path)
        return augment(image, bboxes)

    def eval(self, img_path: str):
        image = self.image(img_path)
        shape = self.shape(image)
        bboxes, diffs = self.label(img_path)
        bboxes = self._relative_to_absolute(bboxes, shape)
        image = self.eval_augment(image, [])[0]
        return (image, self.file_name(img_path), shape, bboxes, diffs)

eval_augment_coco = simple_eval_augment
recover_bboxes_prediction_coco = simple_recover_bboxes_prediction
