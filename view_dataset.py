import os.path as osp

import cv2
import matplotlib.ticker as plticker
import numpy as np
import torch
from matplotlib import pyplot as plt

import tools
from config import cfg
from dataset.train_dataset import TrainDataset, TrainEvalDataset, collate_batch

plt.axis('off')


yaml_file = 'yamls/coco.yaml'
dir_name = 'results/view_datasets'

def draw_bboxes(ax, image, bboxes, prefix=''):
    drawer = tools.BBoxDrawer()
    cnt = 0
    for bbox in bboxes:
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 1e-5:
            continue
        cnt += 1
        if prefix:
            drawer(image, f'{prefix[0]}-{cnt}', bbox, cnt)
        else:
            drawer(image, f'{cnt}', bbox, cnt)
    return cnt

def draw_info(ax, loc, image, indexes, labels, prefix=''):
    s = {
        'Small': 8,
        'Middle': 16,
        'Large':32,
    }[prefix]
    h, w, _ = image.shape
    mask = np.zeros((h//s, w//s), dtype=np.uint8)
    obj_cnt = 0
    for y, x in indexes:
        mask[y, x] = 1
        obj_cnt += 1
    hm = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
    image[hm==1] = (image[hm==1] * 0.25).astype(np.uint8)
    if len(labels):
        bboxes = np.unique(labels[:, :4], axis=0)
    else:
        bboxes = []
    bboxes_cnt = draw_bboxes(ax, image, bboxes, prefix)
    set_ax(ax, loc, f'{prefix}: {bboxes_cnt} bboxes, {obj_cnt} obj', image)

def set_ax(ax, loc, title, image):
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='major', axis='both', linestyle='-')
    ax.title.set_text(title)
    ax.imshow(image)

cfg.merge_from_file(yaml_file)
cfg.train.batch_size = 8
cfg.freeze()
meta_info = tools.build_model(
    cfg.model.cfg_path, device='cpu', dataparallel=False,
)[0].meta_info()
train_dataset = TrainDataset(cfg, meta_info)
# train_dataset = TrainEvalDataset(cfg, meta_info)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.train.batch_size,
    shuffle=False, num_workers=0,
    pin_memory=False, collate_fn=collate_batch,
)
for data in train_dataloader:
    image, index, label, *_ = data
    index = [i.numpy() for i in index]
    label = [l.numpy() for l in label]
    tools.ensure_dir(dir_name)
    image = np.clip(
        (np.transpose(image.numpy(), (0, 2, 3, 1)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255.,
        0, 255,
    ).astype(np.uint8)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    for i, im in enumerate(image):
        h = im.shape[0]
        # bs = np.unique(
        #     np.concatenate([label[j][index[j][0, :] == i, :4] for j in range(3)], axis=0),
        #     axis=0,
        # )
        # im_c = im.copy()
        # draw_bboxes(ax1, im_c, bs)
        ax1.imshow(im)
        st = plticker.MultipleLocator(base=8)
        mt = plticker.MultipleLocator(base=16)
        lt = plticker.MultipleLocator(base=32)
        m = index[0][0, :] == i
        draw_info(ax2, st, im.copy(), index[0][1:3, m].T, label[0][m, :], 'Small')
        m = index[1][0, :] == i
        draw_info(ax3, mt, im.copy(), index[1][1:3, m].T, label[1][m, :], 'Middle')
        m = index[2][0, :] == i
        draw_info(ax4, lt, im.copy(), index[2][1:3, m].T, label[2][m, :], 'Large')
        fig.savefig(osp.join(dir_name, f'{i}.jpg'), bbox_inches='tight', pad_inches=0)
        plt.cla()
    break
