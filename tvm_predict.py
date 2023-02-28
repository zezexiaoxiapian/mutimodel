import numpy as np

import torch
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_runtime as runtime
import tools
import cv2
import time
from numpy_process import preprocess, postprocess

# cfg_path = 'model/cfg/mobilenetv2-fpn.cfg'
cfg_path = 'model/cfg/pruned-mobilenetv2-fpn.cfg'
# weight_path = 'weights/model-59-0.4746.pt'
# weight_path = 'weights/qat/model-79-0.4768.pt'
weight_path = 'weights/model-29-0.4285-quant.pt'
model = tools.build_model(
    cfg_path, weight_path, None, device='cpu', dataparallel=False, onnx=True, quantized=True,
)[0].eval()

log_file = 'myolo.log'
target = tvm.target.cuda()

def compile_tvm_model(model, log_file):
    inp = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    script_module = torch.jit.trace(model, inp).eval()

    # with torch.no_grad():
    #     _ = script_module(inp).numpy()

    input_shapes = [('input', (1, 3, 512, 512))]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))
        return module

def predict(tvm_model, image, in_size):
    ori_size = np.array(image.shape[:2])
    inp = tvm.nd.array(preprocess(image, in_size))
    tvm_model.set_input(input=inp)
    tvm_model.run()
    pred = tvm_model.get_output(0).asnumpy()
    bboxes = postprocess(pred[0], in_size, ori_size)
    return bboxes

# def predict(model, im, in_size):
#     ori_size = np.array(im.shape[:2])
#     inp = torch.from_numpy(preprocess(im, in_size))
#     with torch.no_grad():
#         pred = model(inp).cpu().numpy()
#     bboxes = postprocess(pred[0], in_size, ori_size)
#     return bboxes

if __name__ == '__main__':
    tvm_model = compile_tvm_model(model, log_file)
    in_size = [512, 512]
    image = cv2.imread('data/images/001007.jpg')
    # for _ in range(20):
    # re = predict(model, image, in_size)
    re = predict(tvm_model, image, in_size)
    print(re)
    # t = time.time()
    # for _ in range(200):
    #     re = predict(tvm_model, image, in_size)
    # print((time.time() - t) / 200)
    bdrawer = tools.BBoxDrawer()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in re:
        x1, y1, x2, y2, *_ = [int(n) for n in box]
        cls_idx = int(box[-1])
        text = '{} {:.3f}'.format(str(cls_idx), box[4])
        bdrawer(image, text, (x1, y1, x2, y2), cls_idx)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('results/tvm.jpg', image)