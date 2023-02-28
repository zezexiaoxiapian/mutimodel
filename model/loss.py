import torch
from torch import nn

import tools


def smooth_l1_loss(input, target, beta=1/9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.mean(dim=-1, keepdim=True)

def focal(target, actual, alpha=0.5, gamma=2):
    alpha_t = 2 * torch.abs(target - 1 + alpha)
    focal = alpha_t * torch.pow(torch.abs(target - actual), gamma)
    return focal

class DetectionLoss:

    def __init__(self, meta_info: dict):
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.opt = list(meta_info.values())

    def __call__(self, pred, index, label):
        losses = []
        for p, i, l, o in zip(pred, index, label, self.opt):
            i = i.to(p.device, non_blocking=True)
            l = l.to(p.device, non_blocking=True)
            losses.append(self.loss_per_scale(p, i, l, o))

        xy_loss, obj_loss, cls_loss = 0, 0, 0
        for layer_loss in losses:
            xy_loss += layer_loss[0]
            obj_loss += layer_loss[1]
            cls_loss += layer_loss[2]
        loss_per_branch = [sum(loss) for loss in losses]
        loss = xy_loss + obj_loss + cls_loss

        if torch.isnan(loss):
            print('xy: {}, obj: {}, cls: {}'.format(
                xy_loss.item(), obj_loss.item(), cls_loss.item()
            ))
            raise RuntimeError('NaN in loss')

        return {
            'loss': loss,
            'xy_loss': xy_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'loss_per_branch': loss_per_branch,
        }

    def loss_per_scale(self, pred, index, label, opt):
        xy_loss_gain = 0.05
        obj_loss_gain = 1 * opt['obj_balance']
        cls_loss_gain = 0.05

        num_targets = label.shape[0]
        bs = pred.shape[0]

        xy_loss = 0.
        cls_loss = 0.
        pred_obj = pred[..., 4]
        label_obj = torch.zeros_like(pred_obj, device=pred_obj.device)
        if num_targets:
            img, yc, xc, anchor = index
            respond_pred = pred[img, yc, xc, anchor]
            pred_xy = respond_pred[:, :4]
            pred_cls = respond_pred[:, 5:]
            label_xy = label[:, :4]
            label_cls = label[:, 4:-1]
            label_mixw = label[:, -1:]

            # xy_loss
            giou = tools.giou(pred_xy, label_xy)
            xy_loss = bs * xy_loss_gain * ((1.0 - giou) * label_mixw).mean()

            # obj_loss
            label_obj[img, yc, xc, anchor] = giou.detach().clamp(0.)
            obj_label_mixw = torch.ones_like(pred_obj, device=pred_obj.device)
            obj_label_mixw[img, yc, xc, anchor] = label_mixw[:, 0]

            # cls loss
            cls_loss = bs * cls_loss_gain * \
                (self.bce_loss(pred_cls, label_cls) * label_mixw).mean()
        obj_loss = bs * obj_loss_gain * \
            (self.bce_loss(pred_obj, label_obj) * obj_label_mixw).mean()

        return xy_loss, obj_loss, cls_loss
