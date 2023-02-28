from dataset.voc_sample import VOCSampleGetter, eval_augment_voc, recover_bboxes_prediction_voc
from dataset.visdrone_sample import VisDroneSampleGetter, eval_augment_visdrone, recover_bboxes_prediction_visdrone
from dataset.coco_sample import COCOSampleGetter, eval_augment_coco, recover_bboxes_prediction_coco
from dataset.flir_sample import FLIRSampleGetter, eval_augment_flir, recover_bboxes_prediction_flir

SAMPLE_GETTER_REGISTER = {
    'voc': VOCSampleGetter,
    'visdrone': VisDroneSampleGetter,
    'coco': COCOSampleGetter,
    'flir': FLIRSampleGetter,
}

EVAL_AUGMENT_REGISTER = {
    'voc': eval_augment_voc,
    'visdrone': eval_augment_visdrone,
    'coco': eval_augment_coco,
    'flir': eval_augment_flir,
}

RECOVER_BBOXES_REGISTER = {
    'voc': recover_bboxes_prediction_voc,
    'visdrone': recover_bboxes_prediction_visdrone,
    'coco': recover_bboxes_prediction_coco,
    'flir': recover_bboxes_prediction_flir,
}