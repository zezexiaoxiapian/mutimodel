/* Copyright (c) 1993-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <NvInfer.h>
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <NvOnnxParser.h>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cudawrapper;

static Logger gLogger;

// Number of times we run inference to calculate average time.
constexpr int ITERATIONS = 2;
// Allow TensorRT to use up to 256MB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 28; // 256MB

ICudaEngine *createCudaEngine(string const &onnxModelPath, int batchSize)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig, Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setMaxBatchSize(batchSize);

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 512, 512});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 512, 512});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{8, 3, 512, 512});
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

ICudaEngine *getCudaEngine(string const &onnxModelPath, int batchSize)
{
    string enginePath{getBasename(onnxModelPath) + "_batch" + to_string(batchSize) + ".engine"};
    ICudaEngine *engine{nullptr};

    string buffer = readBuffer(enginePath);

    if (buffer.size())
    {
        // Try to deserialize engine.
        unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }

    if (!engine)
    {
        // Fallback to creating engine from scratch.
        engine = createCudaEngine(onnxModelPath, batchSize);

        if (engine)
        {
            unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    }
    return engine;
}

static int getBindingInputIndex(IExecutionContext *context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext *context, cudaStream_t stream, vector<float> const &inputTensor, vector<float> &outputTensor, void **bindings, int batchSize)
{
    int inputId = getBindingInputIndex(context);

    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void doInference(IExecutionContext *context, cudaStream_t stream, vector<float> const &inputTensor, vector<float> &outputTensor, void **bindings, int batchSize)
{
    CudaEvent start;
    CudaEvent end;
    double totalTime = 0.0;

    for (int i = 0; i < ITERATIONS; ++i)
    {
        float elapsedTime;

        // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
        cudaEventRecord(start, stream);
        launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);
        cudaEventRecord(end, stream);

        // Wait until the work is finished.
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;
    }

    cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
}

struct AffineOpt
{
    array<float, 2> shift;
    float scale;
};

struct BBox
{
    int x1, y1, x2, y2;
    int id;
    int score;
};

ostream &operator<<(ostream &os, const BBox &box)
{
    os << "[ " << box.id << "\t[";
    os << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2;
    return os << " ]\t" << box.score / 1e8f << " ]";
}

typedef vector<BBox> BBoxes;

inline int _affine(float x, float shift, float scale)
{
    return static_cast<int>((x - shift) / scale + 0.5f);
}

vector<BBoxes> filterByConfAndAffine(const vector<float> &out, int nc, float thres, const AffineOpt &ar)
{
    vector<BBoxes> res(nc, BBoxes{});
    for (auto p = out.begin(); p != out.end(); p += 6)
    {
        if (*(p + 5) > thres && *(p + 2) > *p && *(p + 3) > *(p + 1))
        {
            auto clsIdx = static_cast<int>(*(p + 4));
            res[clsIdx].push_back({
                _affine(*p, ar.shift[0], ar.scale),
                _affine(*(p + 1), ar.shift[1], ar.scale),
                _affine(*(p + 2), ar.shift[0], ar.scale),
                _affine(*(p + 3), ar.shift[1], ar.scale),
                clsIdx,
                static_cast<int>(*(p + 5) * 1e8f),
            });
        }
    }
    return res;
}

inline bool byScore(const BBox box1, const BBox box2)
{
    return box1.score > box2.score ? true : false;
}

float iou(const BBox &box1, const BBox &box2)
{
    auto area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    auto area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    auto x1 = max(box1.x1, box2.x1);
    auto y1 = max(box1.y1, box2.y1);
    auto x2 = min(box1.x2, box2.x2);
    auto y2 = min(box1.y2, box2.y2);
    auto w = max(0, x2 - x1 + 1);
    auto h = max(0, y2 - y1 + 1);
    float over_area = w * h;
    return over_area / (area1 + area2 - over_area);
}

BBoxes nms(BBoxes &bboxes, float threshold)
{
    BBoxes results{};
    sort(bboxes.begin(), bboxes.end(), byScore);
    while (bboxes.size() > 0)
    {
        results.push_back(bboxes[0]);
        int index = 1;
        while (index < bboxes.size())
        {
            float iouVal = iou(bboxes[0], bboxes[index]);
            // cout << "iou:" << iouVal << endl;
            if (iouVal > threshold)
                bboxes.erase(bboxes.begin() + index);
            else
                index++;
        }
        bboxes.erase(bboxes.begin());
    }
    return results;
}

pair<cv::Mat, AffineOpt> resizeKeepRatio(cv::Mat image, const cv::Size dsize)
{
    auto [width, height] = image.size();
    auto [twidth, theight] = dsize;
    float reszieRatio = min(
        twidth / static_cast<float>(width),
        theight / static_cast<float>(height));
    int rwidth = reszieRatio * width + 0.5f;
    int rheight = reszieRatio * height + 0.5f;
    int dl = (twidth - rwidth) / 2;
    int du = (theight - rheight) / 2;

    cv::Mat pimg(theight, twidth, CV_8UC3, {128, 128, 128});
    cv::Mat rimg(pimg, cv::Rect(dl, du, rwidth, rheight));
    cv::resize(image, rimg, rimg.size(), 0, 0, cv::INTER_LINEAR);
    return {pimg, {{dl, du}, reszieRatio}};
}

typedef array<float, 3> trif_t;

vector<float> normalize(cv::Mat image, const trif_t std, const trif_t mean)
{
    int w = image.cols;
    int h = image.rows;
    int cs = w * h;
    int elemn = cs * image.channels();
    vector<float> res(elemn);
    image.forEach<cv::Vec3b>(
        [&](const cv::Vec3b &p, const int pos[]) {
            auto bias = pos[0] * h + pos[1];
            res[bias] = ((p[2] / 255.f) - mean[0]) / std[0];
            res[cs + bias] = ((p[1] / 255.f) - mean[1]) / std[1];
            res[2 * cs + bias] = ((p[0] / 255.f) - mean[2]) / std[2];
        });
    return res;
}

int main(int argc, char *argv[])
{
    // Declaring cuda engine.
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    // Declaring execution context.
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
    vector<float> outputTensor;
    void *bindings[2]{0};
    CudaStream stream;

    string imgPath(argv[2]);
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Could not read the image: " << imgPath << "\n";
        return 1;
    }
    auto [pimg, ar] = resizeKeepRatio(img, {512, 512});
    auto inputTensor = normalize(pimg, {0.229, 0.224, 0.225}, {0.485, 0.456, 0.406});
    auto [width, height] = pimg.size();
    cout << width << ", " << height << '\n';
    string onnxModelPath(argv[1]);
    constexpr int batchSize = 1;
    constexpr int numClasses = 20;
    constexpr float confThreshold = 0.3;
    constexpr float nmsThreshold = 0.45;

    // Create Cuda Engine.
    engine.reset(getCudaEngine(onnxModelPath, batchSize));
    if (!engine)
        return 1;

    // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
    assert(engine->getNbBindings() == 2);
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims{engine->getBindingDimensions(i)};
        cout << "Binding " << i << ": ";
        std::for_each(dims.d + 1, dims.d + dims.nbDims, [](int const &v) { cout << v << "x"; });
        cout << "\n";
        size_t size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], batchSize * size * sizeof(float));

        // Resize CPU buffers to fit Tensor.
        if (!engine->bindingIsInput(i))
            outputTensor.resize(size);
    }

    // Create Execution Context.
    context.reset(engine->createExecutionContext());

    Dims dims_i{engine->getBindingDimensions(0)};
    Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
    context->setBindingDimensions(0, inputDims);

    doInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);

    cout << outputTensor[0] << "|" << outputTensor[0 + 6] << "|" << outputTensor[0 + 12] << "\n";
    auto filtered = filterByConfAndAffine(outputTensor, numClasses, confThreshold, ar);
    for (auto &bboxes : filtered)
    {
        auto rb = nms(bboxes, nmsThreshold);
        copy(rb.begin(), rb.end(), ostream_iterator<BBox>(cout, "\n"));
    }
    cout << "\n";

    Dims dims_o{engine->getBindingDimensions(1)};

    for (void *ptr : bindings)
        cudaFree(ptr);

    return 0;
}
