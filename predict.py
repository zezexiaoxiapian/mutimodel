import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import sys

import tools
from config import cfg
from dataset import EVAL_AUGMENT_REGISTER, RECOVER_BBOXES_REGISTER


def main(args):

    def _batch_predict(img_folder):
        tools.ensure_dir(os.path.join(img_folder, 'mark'))
        for fname in tqdm(os.listdir(img_folder)):
            image = cv2.imread(os.path.join(img_folder, fname))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # pylint: disable-msg=not-callable
            original_size = torch.tensor(image.shape[:2], device=device, dtype=torch.float32)
            input_size = torch.tensor([args.size, args.size], device=device, dtype=torch.float32)
            preprocess = EVAL_AUGMENT_REGISTER[dataset_name]([args.size, args.size], device)
            input_image = preprocess(image, [])[0].unsqueeze_(0)
            model.eval()
            with torch.no_grad():
                batch_pred_bbox = model(input_image)
            batch_pred_bbox = RECOVER_BBOXES_REGISTER[dataset_name](batch_pred_bbox, input_size, original_size)

            batch_bboxes = []
            for pred_bboxes in batch_pred_bbox:
                bboxes = tools.torch_nms(
                    pred_bboxes,
                    score_threshold,
                    iou_threshold,
                ).cpu().numpy()
                batch_bboxes.append(bboxes)

            bdrawer = tools.BBoxDrawer()
            for box in batch_bboxes[0]:
                x1, y1, x2, y2, *_ = [int(n) for n in box]
                cls_idx = int(box[-1])
                text = ''
                if not args.no_text:
                    text = '{} {:.3f}'.format(class_names[cls_idx], box[4])
                bdrawer(image, text, (x1, y1, x2, y2), cls_idx)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            args.output = os.path.join(img_folder, 'mark', fname.rsplit('.', 1)[0] + '_mark.jpg')
            cv2.imwrite(args.output, image)

    dataset_name = args.dataset if args.dataset else cfg.dataset.name
    score_threshold = args.threshold
    iou_threshold = args.nms_iou
    class_names = cfg.dataset.classes
    device = torch.device('cpu')

    model = tools.build_model(
        args.cfg, args.weight, None, device=device,
        dataparallel=False, quantized=args.quant, backend=args.backend
    )[0]
    # print(model)

    if args.img_folder:
        _batch_predict(args.img_folder)
        sys.exit(0)

    image = cv2.imread(args.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    the_path = args.img.replace('RGB', 'thermal_8_bit').replace('.jpg', '.jpeg')
    the_img = cv2.imread(the_path, cv2.IMREAD_GRAYSCALE)
    assert the_img is not None, '{} not found'.format(the_path)
    th, tw, *_ = the_img.shape
    image = cv2.resize(image, dsize=(tw, th), interpolation=cv2.INTER_LINEAR)
    bi_image = np.concatenate([image, the_img[..., None]], axis=-1)
    # pylint: disable-msg=not-callable
    original_size = torch.tensor(bi_image.shape[:2], device=device, dtype=torch.float32)
    input_size = torch.tensor([args.size, args.size], device=device, dtype=torch.float32)
    preprocess = EVAL_AUGMENT_REGISTER[dataset_name]([args.size, args.size], device)
    input_image = preprocess(bi_image, [])[0].unsqueeze_(0)
    model.eval()
    radar = input_image[:, 4, :, :].view(input_image.shape[0], -1)
    input_image = input_image[:, :4, :, :]
    with torch.no_grad():
        batch_pred_bbox = model(input_image, radar)
    batch_pred_bbox = RECOVER_BBOXES_REGISTER[dataset_name](batch_pred_bbox, input_size, original_size)
    batch_bboxes = []
    for pred_bboxes in batch_pred_bbox:
        bboxes = tools.torch_nms(
            pred_bboxes,
            score_threshold,
            iou_threshold,
        ).cpu().numpy()
        batch_bboxes.append(bboxes)

    print(f'detect {len(batch_bboxes[0])} objects.')
    print(batch_bboxes[0])

    the_img = cv2.cvtColor(the_img, cv2.COLOR_GRAY2RGB)
    draw_img = the_img
    bdrawer = tools.BBoxDrawer()
    for box in batch_bboxes[0]:
        x1, y1, x2, y2, *_ = [int(n) for n in box]
        cls_idx = int(box[-1])
        text = ''
        if not args.no_text:
            text = '{} {:.3f}'.format(class_names[cls_idx], box[4])
        bdrawer(draw_img, text, (x1, y1, x2, y2), cls_idx)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)

    if not args.output:
        args.output = args.img.rsplit('.', 1)[0] + '_mark.jpg'
    cv2.imwrite(args.output, draw_img)
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('--yaml', default='yamls/voc.yaml', required=False)
    parser.add_argument('--dataset', default='', help='dataset name')
    parser.add_argument('--cfg', default='model/cfg/classifier/resnet50-1g.cfg',help='model cfg file')
    parser.add_argument('--weight', help='model weight')

    parser.add_argument('--quant', help='quantized model', action='store_true', default=False)
    parser.add_argument(
        '--backend', help='quantized backend used in quantized model',
        type=str, default='qnnpack'
    )

    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--nms-iou', help='NMS iou', type=float, default=0.3)
    parser.add_argument('--threshold', help='predict score threshold', type=float, default=0.3)
    parser.add_argument('--img', help='image path', type=str)
    parser.add_argument('--img-folder', help='image folder', type=str, default='')
    parser.add_argument('--output', help='output image path', type=str, default='')

    parser.add_argument('--no-text', help='dont show text on bboxes', action='store_true', default=False)
    args = parser.parse_args()
    if args.yaml:
        cfg.merge_from_file(args.yaml)
    cfg.freeze()
    main(args)
