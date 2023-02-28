from collections import defaultdict, namedtuple
from typing import Callable, Tuple

import numpy as np
import tools
from tools import AverageMeter
import torch
from config import size_fix
from dataset import RECOVER_BBOXES_REGISTER
from dataset.train_dataset import TrainEvalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.loss import DetectionLoss
from model.utils import detection_post_process

AP_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

Label = namedtuple('Label', ['bboxes', 'seen', 'difficult'])

_model_t = Callable[[torch.Tensor], torch.Tensor]

class Evaluator:

    def __init__(self, model: _model_t, dataset: DataLoader, config):
        # super(Evaluator, self).__init__(model)
        self._score_threshold = config.eval.score_threshold
        self._iou_threshold = config.eval.iou_threshold
        self._input_size = size_fix(config.eval.input_size)
        self._recover_bboxes = RECOVER_BBOXES_REGISTER[config.dataset.name.lower()]
        self._classes = config.dataset.classes

        self.model = model
        self.dataset = dataset
        self.init_statics()

        try:
            self.evaluate = self.evaluate_loss if\
                isinstance(self.dataset.dataset, TrainEvalDataset) else self.evaluate_norm
        except:
            self.evaluate = self.evaluate_norm

    def init_statics(self):
        PQ_func = lambda: tools.PriorityQueue(key=lambda x: -x[1][4])
        self.detections_count = 0
        self.detections = defaultdict(PQ_func)
        self.labels = defaultdict(dict)
        self.gt_count = defaultdict(int)

    def evaluate_norm(self) -> tools.AP:
        for data in tqdm(self.dataset):
            img, file_name, img_shape, bbox, diff = data
            radar = img[:, 4, :, :].view(img.shape[0], -1)
            img = img[:, :4, :, :]
            with torch.no_grad():
                pred_bbox = self.model(img, radar)
            self.add_batch(pred_bbox, file_name, img_shape, bbox, diff)
        return self.AP()

    def evaluate_loss(self) -> Tuple[tools.AP, dict]:
        meta_info = tools.bare_model(self.model).meta_info()
        criterion = DetectionLoss(meta_info)
        loss_dict = {
            'loss': AverageMeter(),
            'xy_loss': AverageMeter(),
            'obj_loss': AverageMeter(),
            'cls_loss': AverageMeter(),
        }
        for data in tqdm(self.dataset):
            img, index, label, file_name, img_shape, bbox, diff = data
            radar = img[:, 4, :, :].view(img.shape[0], -1)
            img = img[:, :4, :, :]
            with torch.no_grad():
                pred = self.model(img, radar)
                losses = criterion(pred, index, label)
                for k, v in loss_dict.items():
                    v.update(losses[k].item(), n=img.shape[0])
            pred_bbox = detection_post_process(pred)
            self.add_batch(pred_bbox, file_name, img_shape, bbox, diff)
        loss_dict = {k: v.get_avg_reset() for k, v in loss_dict.items()}
        return self.AP(), loss_dict

    def AP(self) -> tools.AP:
        num_classes, num_iou_thres = len(self._classes), len(AP_IOU_THRESHOLDS)
        AP_class_iou = np.zeros((num_classes, num_iou_thres))
        metrics = tools.AP(np.zeros(num_iou_thres), np.zeros(num_classes), 0., AP_class_iou, self._classes, AP_IOU_THRESHOLDS)
        process_bar = tqdm(total=self.detections_count)
        for class_index, detections in self.detections.items():
            tp = np.zeros((len(AP_IOU_THRESHOLDS), len(detections)))
            fp = np.zeros((len(AP_IOU_THRESHOLDS), len(detections)))
            for detect_index, (file_name, bbox) in enumerate(detections):
                label = self.labels[file_name].get(class_index)
                if label is None:
                    fp[:, detect_index] = 1
                    process_bar.update()
                    continue
                BBGT = label.bboxes
                bb = bbox[:4]
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                # iou_max = np.max(overlaps)
                # j_max = np.argmax(overlaps)
                # ious = tools.iou_calc1(bbox[:4], label.bboxes)
                # iou_max = np.max(ious)
                # j_max = np.argmax(ious)
                for iou_index, iou_threshold in enumerate(AP_IOU_THRESHOLDS):
                    pick_index = -1
                    pick_iou = min(iou_threshold, 1 - 1e-10)
                    for match_index, match_iou in enumerate(overlaps):
                        if label.seen[iou_index, match_index]:
                            continue
                        if pick_index > -1 and not label.difficult[pick_index] and\
                            label.difficult[match_index]:
                            break
                        if match_iou < pick_iou:
                            continue
                        pick_index = match_index
                        pick_iou = match_iou
                    if label.difficult[pick_index]:
                        continue
                    if pick_index == -1 or label.seen[iou_index, pick_index]:
                        fp[iou_index, detect_index] = 1
                        continue
                    tp[iou_index, detect_index] = 1
                    label.seen[iou_index, pick_index] = True
                # iou_mask = iou_max > AP_IOU_THRESHOLDS
                # for iou_index, ok in enumerate(iou_mask):
                #     if ok:
                #         if label.difficult[j_max]:
                #             continue
                #         if label.seen[iou_index, j_max]:
                #             fp[iou_index, detect_index] = 1
                #         else:
                #             tp[iou_index, detect_index] = 1
                #             label.seen[iou_index, j_max] = True
                #     else:
                #         fp[iou_index, detect_index] = 1
                process_bar.update()
            fp = np.cumsum(fp, axis=1)
            tp = np.cumsum(tp, axis=1)
            rec = tp / self.gt_count[class_index]
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            AP_class_iou[class_index] = self.calculate_ap_by_recall_precision(rec, prec)
            APs = np.mean(AP_class_iou, axis=1)
            mAPs = np.mean(AP_class_iou, axis=0)
            metrics = tools.AP(mAPs, APs, np.mean(mAPs), AP_class_iou, self._classes, AP_IOU_THRESHOLDS)
        process_bar.close()
        self.init_statics()
        return metrics

    @staticmethod
    def calculate_ap_by_recall_precision(recs: np.ndarray, precs: np.ndarray) -> float:
        mrecs = np.pad(recs, ((0, 0), (1, 1)), constant_values=(0., 1.))
        mpres = np.pad(precs, ((0, 0), (1, 1)), constant_values=0.)

        # use built-in list is faster than numpy array here
        mpres = mpres.tolist()
        lmpre = len(mpres[0])
        for mpre in mpres:
            for i in range(lmpre - 1, 0, -1):
                if mpre[i] > mpre[i-1]:
                    mpre[i-1] = mpre[i]
        mpres = np.array(mpres)

        ap = np.sum(np.diff(mrecs) * mpres[:, 1:], axis=1)
        return ap

    def add_batch(self, pred_bbox, file_name, img_shape, gt_bbox, diff):
        device = pred_bbox.device
        input_size = torch.FloatTensor(self._input_size).to(device)
        img_shape = img_shape.to(device)
        pred_bbox = self._recover_bboxes(pred_bbox, input_size, img_shape)

        for file_name, gt_bboxes, diffs, pred_bboxes\
            in zip(file_name, gt_bbox, diff, pred_bbox):
            bboxes = tools.torch_nms(
                pred_bboxes,
                self._score_threshold,
                self._iou_threshold
            ).cpu().numpy()
            print(file_name)
            self.add_detections(file_name, bboxes)

            self.add_labels(file_name, gt_bboxes, diffs)

    def add_detections(self, file_name: str, bboxes: np.ndarray):
        self.detections_count += len(bboxes)
        for bbox in bboxes:
            self.detections[int(bbox[-1])].push((file_name, bbox))

    def add_labels(self, file_name: str, bboxes: np.ndarray, diffs: np.ndarray):
        if bboxes.size == 0: return
        classes = bboxes[:, -1].astype(int)
        for class_index in set(classes):
            select_indeces = classes == class_index
            select_bboxes = bboxes[select_indeces][:, :4]
            select_diffs = diffs[select_indeces].astype(np.bool)
            diffs_perm = np.argsort(select_diffs)
            select_bboxes = select_bboxes[diffs_perm]
            select_diffs = select_diffs[diffs_perm]
            seens = np.zeros((len(AP_IOU_THRESHOLDS), len(select_bboxes))).astype(np.bool)
            self.labels[file_name][class_index] = Label(select_bboxes, seens, select_diffs)
            self.gt_count[class_index] += np.sum(~select_diffs)
