import numpy as np

import torch
import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
import tools
import os
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_52')

def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_runtime.GraphModule(lib["default"](tvm.context(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).asnumpy(), runtime

cfg_path = ''
# weight_path = 'weights/model-59-0.4746.pt'
# weight_path = 'weights/qat/model-79-0.4768.pt'
weight_path = 'weights/model-89-0.3512.pt'
model = tools.build_model(
    cfg_path, weight_path, None, device='cpu', dataparallel=False, onnx=True, quantized=False
)[0].eval()
# weight_path = 'weights/model-79-0.4768.pt'
# model = tools.build_model(
#     cfg_path, weight_path, None, device='cpu', dataparallel=False, quantized=True, onnx=True,
# )[0].eval()

inp = torch.randn(1, 5, 512, 512)
radar = inp[:, 4, :, :].view(inp.shape[0], -1)
inp = inp[:, :4, :, :]
script_module = torch.jit.trace(model, [inp, radar]).eval()

with torch.no_grad():
    pt_result = script_module(inp, radar).numpy()

input_name = "input"
input_shapes = [(input_name+'1', (1, 4, 512, 512)), (input_name+'2', (1, 512 * 512))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
# print(mod)

# tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target="cuda")

# n_repeat = 100  # should be bigger to make the measurement more accurate
# ctx = tvm.gpu(0)
# ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=n_repeat)
# prof_res = np.array(ftimer().results) * 1e3
# print("Elapsed average ms:", np.mean(prof_res))

log_file = 'results/tvm_log/pcspnet-pan-980ti.log'
target = tvm.target.cuda(model='tx2')
target_host = "llvm -mtriple=aarch64-linux-gnu"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
            'nx',
            host="0.0.0.0",
            port=9190,
            number=10, timeout=5, min_repeat_ms=150
        ),
    ),
}

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in list(enumerate(reversed(tasks)))[39:]:
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    # tasks = autotvm.task.extract_from_program(
    #     mod["main"], target=target, target_host=target_host, params=params, ops=(relay.op.get("nn.conv2d"),)
    # )

    # run tuning tasks
    print("Tuning...")
    # tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, target_host=target_host, params=params)

        lib.export_library('export/flir-pcpnet-pan-980ti-aarch64-tx2.so')
        # load parameters
        ctx = tvm.gpu(0) if str(target) == 'gpu' else tvm.cpu()
        module = runtime.GraphModule(lib["default"](ctx))
        data_tvm1 = tvm.nd.array((np.random.uniform(size=(1, 4, 512, 512))).astype(np.float32))
        data_tvm2 = tvm.nd.array((np.random.uniform(size=(1, 512 * 512))).astype(np.float32))
        module.set_input(input_name+'1', data_tvm1)
        module.set_input(input_name+'2', data_tvm2)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

tune_and_evaluate(tuning_option)
