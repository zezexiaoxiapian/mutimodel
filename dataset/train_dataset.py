import random
from math import ceil

import numpy as np
import torch
from config import size_fix, sizes_fix
from torch.utils.data import Dataset

from dataset import SAMPLE_GETTER_REGISTER
from dataset.utils import cat_if, read_txt_file


def collate_batch(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, np.ndarray):
        return collate_batch([torch.as_tensor(b) for b in batch])
    else:
        # batch_index: [batch_size, 3, 4, num_targets]
        # batch_label: [batch_size, 3, num_targets, 5]
        # others 在 TrainEvalDataset 时存放[file_name, shape, bboxes, diffs]
        batch_image, batch_index, batch_label, *others = zip(*batch)
        if others:
            others[1] = torch.as_tensor(others[1])
        for i, index in enumerate(batch_index):
            for scale in index:
                if len(scale): scale[0, :] = i
        # batch_index: [3, 4, num_targets_all]
        batch_index = [collate_batch(cat_if(scale_index, axis=-1))
            for scale_index in zip(*batch_index)]
        # batch_label: [3, num_targets_all, 5]
        batch_label = [collate_batch(cat_if(scale_label, axis=0))
            for scale_label in zip(*batch_label)]
        return (collate_batch(batch_image), batch_index, batch_label, *others)

def create_label(bboxes, input_size, num_classes, anchors, strides):
    # bboxes: [num_targets, 6(x1, y1, x2, y2, class_index, mixup_weight)]
    # input_size: [2(h, w)]
    # num_classes: int
    # anchors: [num_anchors, 2(w, h)]
    # strides: [num_yolo_layers]
    index = [list() for i in range(3)]
    label = [list() for i in range(3)]

    offsets = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    out_sizes = (input_size // strides[:, None])[:, [1, 0]]
    for bbox in bboxes:
        # bbox: [6] = [x1, y1, x2, y2, class_index, mixup_weight]
        # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽
        bbox_coor = bbox[:4]
        bbox_class_ind = int(bbox[4])
        bbox_mixw = bbox[5]
        # bbox_xywh: [xc, yc, w, h]
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                    bbox_coor[2:] - bbox_coor[:2]], axis=-1)

        # label smooth
        onehot = np.zeros(num_classes, dtype=np.float32)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        # 计算当前 bbox 对所有 anchor 长宽的比值
        # wh_ratio: [num_anchors, 2] = [9, 2]
        wh_ratio = np.clip(bbox_xywh[2:] / anchors, 1e-16, np.inf)
        # 只有当当前 bbox 所有长宽比及其倒数都小于 4，才匹配该 anchor
        # anchor_mask: [9] = [num_anchors]
        anchor_mask = np.max([wh_ratio, 1./wh_ratio], axis=2).max(0) < 4.
        if not anchor_mask.any():
            # print('dataset: all anchors missed!, shape {}'.format(bbox_xywh[2:]))
            continue
        # 在不同尺度缩放下的中心点位置
        # xcyc_indexes: [yolo层数量(3), 2]
        xcyc_indexes = bbox_xywh[:2] / strides[:, None]
        # 中心点位置上下左右四个格子是否满足要求，即中心点坐标偏向哪个方向
        leftup_mask = (xcyc_indexes % 1. < 0.5) & (xcyc_indexes > 1.)
        rightdown_mask = (xcyc_indexes % 1. > 0.5) & (xcyc_indexes < (out_sizes - 1.))
        # lurd_mask: [yolo层数量(3), 4]
        lurd_mask = np.concatenate([leftup_mask, rightdown_mask], axis=-1)

        for i in anchor_mask.nonzero()[0]:
            scale_branch, ratio_branch = i // 3, i % 3
            xcyc = (xcyc_indexes[scale_branch]).astype(np.int32)
            # all_xcyc: [num_targets, 2]
            all_xcyc = np.concatenate([
                xcyc[None, :],
                xcyc + offsets[lurd_mask[scale_branch]],
            ], axis=0)
            num_targets = all_xcyc.shape[0]
            # all_indexes: [4(img_index, yc, xc, anchor_index), num_targets]
            all_indexes = np.concatenate([
                np.zeros((num_targets, 1), dtype=np.int32),
                all_xcyc[:, [1, 0]],
                np.full((num_targets, 1), ratio_branch, dtype=np.int32),
            ], axis=-1).T
            # all_labels: [num_targets, num_classes + 5]
            all_labels = np.tile(
                np.concatenate([bbox_coor, smooth_onehot, [bbox_mixw]], axis=-1),
                (num_targets, 1),
            )
            index[scale_branch].append(all_indexes)
            label[scale_branch].append(all_labels)

    index = [cat_if(i, axis=-1) for i in index]
    label = [cat_if(l, axis=0) for l in label]
    return index, label

class TrainDataset(Dataset):

    def __init__(self, config, meta_info: dict):
        self._dataset_name = config.dataset.name.lower()
        self._dataset_file = config.dataset.train_txt_file
        self._input_sizes = sizes_fix(config.train.input_sizes)
        self._strides = np.array(list(meta_info.keys()))
        self._batch_size = config.train.batch_size
        self._classes = config.dataset.classes
        self._num_classes = len(self._classes)
        self._anchors = np.array(
            [mi['anchors'] for mi in meta_info.values()], dtype=np.float32,
        ).reshape(-1, 2)
        self._anchor_per_scale = len(self._anchors[0])

        self._imgs = read_txt_file(self._dataset_file)
        self._num_imgs = len(self._imgs)

        self.sample_getter = SAMPLE_GETTER_REGISTER[self._dataset_name](
            mode='train', classes=self._classes
        ).set_train_augment(
            config.augment, self._get_input_size, self.sample_img_path
        )

        self.init_shuffle()

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._num_imgs

    def init_shuffle(self):
        batch_len = ceil(self._num_imgs / self._batch_size)
        self._length = batch_len * self._batch_size
        self._shuffle_indexes = random.choices(range(self._num_imgs), k=self._length)
        self._shuffle_sizes = random.choices(self._input_sizes, k=batch_len)
        max_index = np.argmax([h*w for h, w in self._input_sizes])
        self._shuffle_sizes[0] = self.input_size = self._input_sizes[max_index]

    def _get_input_size(self):
        return self.input_size

    def sample_img_path(self):
        idx = random.randint(0, self._num_imgs - 1)
        return self._imgs[idx]

    def __getitem__(self, index):
        self.input_size = self._shuffle_sizes[index // self._batch_size]

        img_name = self._imgs[self._shuffle_indexes[index]]
        image, bboxes = self.sample_getter(img_name)
        labels = create_label(
            bboxes, self.input_size, self._num_classes, self._anchors, self._strides,
        )
        return (image, *labels)

class TrainEvalDataset(Dataset):

    def __init__(self, config, meta_info: dict):
        self._dataset_name = config.dataset.name.lower()
        self._dataset_file = config.dataset.eval_txt_file
        self._input_size = size_fix(config.eval.input_size)
        self._strides = np.array(list(meta_info.keys()))
        self._batch_size = config.eval.batch_size
        self._classes = config.dataset.classes
        self._num_classes = len(self._classes)
        self._anchors = np.array(
            [mi['anchors'] for mi in meta_info.values()], dtype=np.float32,
        ).reshape(-1, 2)
        self._anchor_per_scale = len(self._anchors[0])

        self._imgs = read_txt_file(self._dataset_file)
        self._num_imgs = len(self._imgs)

        self.sample_getter = SAMPLE_GETTER_REGISTER[self._dataset_name](
            mode='eval', classes=self._classes
        ).set_eval_augment(self._input_size)

        batch_len = ceil(self._num_imgs / self._batch_size)
        self._length = batch_len * self._batch_size

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._num_imgs

    def __getitem__(self, index):
        if index >= self._num_imgs: raise StopIteration
        img_name = self._imgs[index]
        image, file_name, shape, bboxes, diffs = self.sample_getter(img_name)
        th, tw = self._input_size
        oh, ow = shape
        r = min(tw / ow, th / oh)
        dl = (tw - round(r * ow)) // 2
        du = (th - round(r * oh)) // 2
        bboxes_affine = bboxes.copy()
        bboxes_affine[:, [0, 2]] = bboxes_affine[:, [0, 2]] * r + dl
        bboxes_affine[:, [1, 3]] = bboxes_affine[:, [1, 3]] * r + du
        # 在 bboxes 添加一维 mixup weight
        bboxes_mixup = np.concatenate(
            [bboxes_affine, np.ones((bboxes.shape[0], 1), dtype=np.float32)], axis=-1,
        )
        labels = create_label(
            bboxes_mixup, self._input_size, self._num_classes, self._anchors, self._strides,
        )
        return (image, *labels, file_name, shape, bboxes, diffs)
