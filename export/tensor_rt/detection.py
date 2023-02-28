import os
import time

import cv2
import numpy as np
import tensorrt as trt
from tqdm import tqdm
import random

from export.tensor_rt import common

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_VAR = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALPHA = 0.75
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.
    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.
    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img

def py_cpu_nms(dets, thresh):
    x1 = dets[..., 0]
    y1 = dets[..., 1]
    x2 = dets[..., 2]
    y2 = dets[..., 3]
    scores = dets[..., 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def get_converter(image, net_size, iou_thres, conf_thres, num_classes):
    img_h, img_w = image.shape[:2]
    target_h, target_w = net_size
    resize_ratio = min(target_w / img_w, target_h / img_h)
    resize_w = round(resize_ratio * img_w)
    resize_h = round(resize_ratio * img_h)
    dl = (target_w - resize_w) // 2
    dr = target_w - resize_w - dl
    du = (target_h - resize_h) // 2
    dd = target_h - resize_h - du

    def preprocess(img):
        image_resized = cv2.resize(img, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        img_padded = np.pad(
            image_resized, ((du, dd), (dl, dr), (0, 0)),
            'constant', constant_values=0.
        )
        img_padded = img_padded.astype(np.float32, copy=False)
        img_normed = (img_padded/255. - IMG_MEAN) / IMG_VAR
        img_bchw = np.transpose(img_normed, (2, 0, 1))[None, ...]
        return img_bchw

    def postprocess(bboxes: np.array):
        bboxes = bboxes.reshape((-1, 6))
        bboxes = bboxes[bboxes[..., 5] > conf_thres]
        bboxes[..., :4] = (bboxes[..., :4] - [dl, du, dl, du]) / resize_ratio
        classes_index = bboxes[..., 4].astype(np.int)
        remain_bboxes = []
        for i in range(num_classes):
            dets = bboxes[i == classes_index, :]
            remain = dets[py_cpu_nms(dets, iou_thres)]
            remain_bboxes.append(remain)
        return remain_bboxes

    return preprocess, postprocess

# onnx_file_path = 'export/myolo.onnx'
# engine_file_path = 'export/myolo.trt'

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(onnx_file_path, engine_file_path="", shape=(512, 512)):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # builder.strict_type_constraints = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, *shape]
            for layer in network:
                print(layer.type, layer.get_output(0).shape)
            assert 0
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

# trt_outputs = []
# with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
#     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
#     # Do inference
#     # print('Running inference on image {}...'.format(input_image_path))
#     # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
#     image = np.random.rand(1, 3, 512, 512)
#     image = np.array(image, dtype=np.float32, order='C')
#     inputs[0].host = image
#     for _ in range(10):
#         trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#     n = 100
#     start = time.time()
#     for _ in tqdm(range(n)):
#         trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#     print('{} fps'.format(n / (time.time() - start)))

def gen_colors(num):
    return [random.choices(range(256), k=3) for _ in range(num)]

class Detection:

    def __init__(self, engine_file_path, classes=None, net_size=(512, 512), iou_threshold=0.45, conf_threshold=0.3):
        self.classes = CLASSES if classes is None else classes
        self.net_size = net_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        self.palette = np.random.RandomState(123).randint(0, 256, (256, 3), dtype=np.uint8)

        self.engine = get_engine('export/myolo.onnx', engine_file_path, net_size)
        self.context = self.engine.create_execution_context()
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.pre_fn, self.post_fn = None, None

    def predict(self, image: np.array):
        if self.pre_fn is None:
            self.pre_fn, self.post_fn = get_converter(
                image, self.net_size, self.iou_threshold, self.conf_threshold, len(self.classes),
            )

        self.inputs[0].host = np.ascontiguousarray(self.pre_fn(image))
        trt_outputs = common.do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream
        )
        classed_bboxes = self.post_fn(trt_outputs[0])
        image = image.copy()
        for i, (cname, bboxes) in enumerate(zip(CLASSES, classed_bboxes)):
            for box in bboxes:
                x1, y1, x2, y2, *_ = [int(n) for n in box]
                color = [int(v) for v in self.palette[i]]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text_loc = (max(x1+2, 0), max(y1+2, 0))
                text = '{} {:.2f}'.format(cname, box[5])
                draw_boxed_text(image, text, text_loc, color)
        return image

    def close(self):
        del self.outputs
        del self.inputs
        del self.stream


if __name__ == "__main__":
    n = 200
    d = Detection('export/myolo.trt')
    image = cv2.imread('data/images/ajust_image_hsv.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.time()
    for _ in range(20):
        __ = d.predict(image)
    for _ in tqdm(range(n)):
        rimg = d.predict(image)
    elaps = time.time() - start
    print('{} FPS'.format(n / elaps))
    rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)
    cv2.imwrite('data/images/0422174423_trt_mark.jpg', rimg)
    d.close()