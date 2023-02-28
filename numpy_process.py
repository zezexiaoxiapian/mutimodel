import numpy as np
import cv2


def preprocess(image, in_size, pad_val=128):
    th, tw = in_size
    img_h, img_w = image.shape[:2]
    r = min(tw / img_w, th / img_h)
    resize_w = round(r * img_w)
    resize_h = round(r * img_h)
    image = cv2.resize(image, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    dl = (tw - resize_w) // 2
    dr = tw - resize_w - dl
    du = (th - resize_h) // 2
    dd = th - resize_h - du
    image = np.pad(
        image, ((du, dd), (dl, dr), (0, 0)),
        'constant', constant_values=pad_val,
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32, copy=False)
    mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image/255. - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return image

def postprocess(pred, in_size, ori_size, conf_thres=0.4, iou_thres=0.5, max_wh=4096, max_dets=300):
    # in_size and ori_size in [H, W]
    pred = pred[pred[:, 4] > conf_thres]
    if not pred.shape[0]:
        return np.array([])

    pred[:, 5:] *= pred[:, 4:5]
    i, j = (pred[:, 5:] > conf_thres).nonzero()
    box = pred[:, :4]
    pred = np.concatenate((box[i], pred[i, j+5, None], j[:, None].astype(np.float32)), axis=1)
    if not pred.shape[0]:
        return np.array([])

    r = np.min(in_size / ori_size)
    delta = ((in_size - (r * ori_size)) // 2)[[1, 0, 1, 0]]
    pred[:, :4] -= delta
    pred[:, :4] /= r
    np.clip(pred[:, :4], 0, ori_size[[1, 0, 1, 0]], out=pred[:, :4])

    boxes, scores = pred[:, :4] + j[:, None] * max_wh, pred[:, 4]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = [] # 保留的结果框集合
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i) # 保留该类剩余box中得分最高的一个
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        over = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的矩形框索引
        inds = np.where(over <= iou_thres)[0]
        # 将order序列更新
        order = order[inds + 1]

    return pred[keep]
