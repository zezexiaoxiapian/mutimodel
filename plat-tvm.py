from itertools import cycle
from time import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tvm
from tvm.contrib import graph_executor

import tools
from config import cfg
from dataset import EVAL_AUGMENT_REGISTER
from numpy_process import postprocess
from dataset.utils import read_txt_file
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_52')




cfg.merge_from_file('yamls/flir.yaml')
cfg.freeze()

dataset_name = cfg.dataset.name
score_threshold = 0.4
iou_threshold = 0.6
class_names = cfg.dataset.classes

lib = tvm.runtime.load_module('export/flir-pcpnet-pan-980ti-aarch64-tx2.so')
ctx = tvm.gpu(0)
module = graph_executor.GraphModule(lib["default"](ctx))

fig, axs = plt.subplots(2, 2)
plt.ion()

img_paths = read_txt_file(cfg.dataset.eval_txt_file)
for i, img_path in enumerate(cycle(img_paths)):
    ori_image = cv2.imread(img_path)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    the_path = img_path.replace('RGB', 'thermal_8_bit').replace('.jpg', '.jpeg')
    the_img = cv2.imread(the_path, cv2.IMREAD_GRAYSCALE)
    assert the_img is not None, '{} not found'.format(the_path)
    th, tw, *_ = the_img.shape
    image = cv2.resize(ori_image, dsize=(tw, th), interpolation=cv2.INTER_LINEAR)
    bi_image = np.concatenate([image, the_img[..., None]], axis=-1)

    label_path = img_path.replace('RGB', 'txt').replace('.jpg', '.txt')
    bbs = []
    fr = open(label_path, 'r')
    for line in fr.readlines():
        ann = line.split(' ')
        cls_idx = int(ann[0])
        half_rw, half_rh = float(ann[3]) / 2, float(ann[4]) / 2
        rx1 = float(ann[1]) - half_rw
        ry1 = float(ann[2]) - half_rh
        rx2 = float(ann[1]) + half_rw
        ry2 = float(ann[2]) + half_rh
        box = [rx1, ry1, rx2, ry2, cls_idx]
        bbs.append(box)
    fr.close()
    bbs = np.array(bbs, dtype=np.float32)
    bbs[:, :-1] *= np.tile((tw, th), 2)

    # pylint: disable-msg=not-callable
    original_size = np.array(image.shape[:2])
    input_size = [512, 512]
    preprocess = EVAL_AUGMENT_REGISTER[dataset_name](input_size, 'cpu')
    preprocess.transforms = preprocess.transforms[:-1]

    input_image = preprocess(bi_image, bbs)[0]
    input_image = np.transpose(input_image, (2, 0, 1)).astype(np.float32)[None, ...]
    radar_numpy = input_image[:, 4, :, :].reshape(input_image.shape[0], -1)
    radar = tvm.nd.array(radar_numpy)
    input_image = tvm.nd.array(input_image[:, :4, :, :])
    start = time()
    module.set_input('input1', input_image)
    module.set_input('input2', radar)
    module.run()
    pred = module.get_output(0).asnumpy()
    elapse = (time() - start) * 1000
    bboxes = postprocess(pred[0], input_size, original_size)
    fig.suptitle(img_path+'\nLatency: {:.2f}ms'.format(elapse))

    print(f'detect {len(bboxes)} objects.')

    the_img = cv2.cvtColor(the_img, cv2.COLOR_GRAY2RGB)
    draw_img = the_img.copy()
    bdrawer = tools.BBoxDrawer()
    for box in bboxes:
        x1, y1, x2, y2, *_ = [int(n) for n in box]
        cls_idx = int(box[-1])
        text = '{} {:.3f}'.format(class_names[cls_idx], box[4])
        bdrawer(draw_img, text, (x1, y1, x2, y2), cls_idx)
    # draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1200, 900)
    # cv2.imshow('image', draw_img)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    if i == 0:
        ax1 = axs[0, 0].imshow(ori_image)
        ax2 = axs[0, 1].imshow(the_img)
        ax3 = axs[1, 0].imshow(radar_numpy.reshape(512, 512))
        ax4 = axs[1, 1].imshow(draw_img)
    else:
        ax1.set_data(ori_image)
        ax2.set_data(the_img)
        ax3.set_data(radar_numpy.reshape(512, 512))
        ax4.set_data(draw_img)
    fig.canvas.draw()
    plt.pause(0.01)
    fig.canvas.flush_events()
    #plt.savefig('results/flir/' + '%d.png' % i)
