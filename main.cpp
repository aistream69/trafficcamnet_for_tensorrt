#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "common.h"
#include "argparse.h"
#include "nvdsinfer.h"
#include "nvdsinfer_context.h"
#include "nvdsinfer_custom_impl.h"
#include "log.h"

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

using namespace nvinfer1;

static uint32_t m_NumDetectedClasses = 4;
static std::vector<std::vector<std::string>> m_Labels;

std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex,
                                       std::vector<NvDsInferParseObjectInfo>& bbox,
                                       const float nmsThreshold) {
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(bbox.at(idx), bbox.at(kept_idx));
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

void filterTopKOutputs(const int topK,
           std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if(topK < 0)
        return;

    std::stable_sort(objectList.begin(), objectList.end(),
                    [](const NvDsInferObjectDetectionInfo& obj1, const NvDsInferObjectDetectionInfo& obj2) {
                        return obj1.detectionConfidence > obj2.detectionConfidence; });
    objectList.resize(static_cast<size_t>(topK) <= objectList.size() ? topK : objectList.size());
}

void clusterAndFillDetectionOutputNMS(
            std::vector<std::vector<NvDsInferObjectDetectionInfo>>& m_PerClassObjectList,
            std::vector<NvDsInferDetectionParams>& m_PerClassDetectionParams,
            NvDsInferDetectionOutput &output) {
    auto maxComp = [](const std::vector<NvDsInferObjectDetectionInfo>& c1,
                    const std::vector<NvDsInferObjectDetectionInfo>& c2) -> bool
                    { return c1.size() < c2.size(); };

    std::vector<std::pair<float, int>> scoreIndex;
    std::vector<NvDsInferObjectDetectionInfo> clusteredBboxes;
    auto maxElement = *std::max_element(m_PerClassObjectList.begin(),
                            m_PerClassObjectList.end(), maxComp);
    clusteredBboxes.reserve(maxElement.size() * m_NumDetectedClasses);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        if(!m_PerClassObjectList[c].empty())
        {
            scoreIndex.reserve(m_PerClassObjectList[c].size());
            scoreIndex.clear();
            for (size_t r = 0; r < m_PerClassObjectList[c].size(); ++r)
            {
                scoreIndex.emplace_back(std::make_pair(m_PerClassObjectList[c][r].detectionConfidence, r));
            }
            std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
                            [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                                return pair1.first > pair2.first; });
            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(scoreIndex, m_PerClassObjectList[c],
                            m_PerClassDetectionParams[c].nmsIOUThreshold);

            std::vector<NvDsInferObjectDetectionInfo> postNMSBboxes;
            for(auto idx : indices) {
                if(m_PerClassObjectList[c][idx].detectionConfidence >
                m_PerClassDetectionParams[c].postClusterThreshold)
                {
                    postNMSBboxes.emplace_back(m_PerClassObjectList[c][idx]);
                }
            }
            filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, postNMSBboxes);
            clusteredBboxes.insert(clusteredBboxes.end(),postNMSBboxes.begin(), postNMSBboxes.end());
        }
    }

    output.objects = new NvDsInferObject[clusteredBboxes.size()];
    output.numObjects = 0;

    for(uint i=0; i < clusteredBboxes.size(); ++i)
    {
        NvDsInferObject &object = output.objects[output.numObjects];
        object.left = clusteredBboxes[i].left;
        object.top = clusteredBboxes[i].top;
        object.width = clusteredBboxes[i].width;
        object.height = clusteredBboxes[i].height;
        object.classIndex = clusteredBboxes[i].classId;
        object.label = nullptr;
        object.mask = nullptr;
        if (object.classIndex < static_cast<int>(m_Labels.size()) && m_Labels[object.classIndex].size() > 0)
                object.label = strdup(m_Labels[object.classIndex][0].c_str());
        object.confidence = clusteredBboxes[i].detectionConfidence;
        output.numObjects++;
    }
}

void preClusteringThreshold(
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    objectList.erase(std::remove_if(objectList.begin(), objectList.end(),
               [detectionParams](const NvDsInferObjectDetectionInfo& obj)
               { return (obj.classId >= detectionParams.numClassesConfigured) ||
                        (obj.detectionConfidence <
                        detectionParams.perClassPreclusterThreshold[obj.classId])
                        ? true : false;}),objectList.end());
}

bool parseBoundingBox(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                      NvDsInferNetworkInfo const& networkInfo,
                      NvDsInferParseDetectionParams const& detectionParams,
                      std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  int outputCoverageLayerIndex = -1;
  int outputBBoxLayerIndex = -1;

  for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
  {
      if (strstr(outputLayersInfo[i].layerName, "bbox") != nullptr)
      {
          outputBBoxLayerIndex = i;
      }
      if (strstr(outputLayersInfo[i].layerName, "cov") != nullptr)
      {
          outputCoverageLayerIndex = i;
      }
  }

  if (outputCoverageLayerIndex == -1)
  {
      printf("Could not find output coverage layer for parsing objects\n");
      return false;
  }
  if (outputBBoxLayerIndex == -1)
  {
      printf("Could not find output bbox layer for parsing objects\n");
      return false;
  }

  float *outputCoverageBuffer =
      (float *)outputLayersInfo[outputCoverageLayerIndex].buffer;
  float *outputBboxBuffer =
      (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;

  NvDsInferDimsCHW outputCoverageDims;
  NvDsInferDimsCHW outputBBoxDims;

  getDimsCHWFromDims(outputCoverageDims,
      outputLayersInfo[outputCoverageLayerIndex].inferDims);
  getDimsCHWFromDims(
      outputBBoxDims, outputLayersInfo[outputBBoxLayerIndex].inferDims);

  unsigned int targetShape[2] = { outputCoverageDims.w, outputCoverageDims.h };
  float bboxNorm[2] = { 35.0, 35.0 };
  float gcCenters0[targetShape[0]];
  float gcCenters1[targetShape[1]];
  int gridSize = outputCoverageDims.w * outputCoverageDims.h;
  int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, outputBBoxDims.w);
  int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, outputBBoxDims.h);

  for (unsigned int i = 0; i < targetShape[0]; i++)
  {
      gcCenters0[i] = (float)(i * strideX + 0.5);
      gcCenters0[i] /= (float)bboxNorm[0];
  }
  for (unsigned int i = 0; i < targetShape[1]; i++)
  {
      gcCenters1[i] = (float)(i * strideY + 0.5);
      gcCenters1[i] /= (float)bboxNorm[1];
  }

  unsigned int numClasses =
      std::min(outputCoverageDims.c, detectionParams.numClassesConfigured);
  for (unsigned int classIndex = 0; classIndex < numClasses; classIndex++)
  {

      float *outputX1 = outputBboxBuffer
          + classIndex * sizeof (float) * outputBBoxDims.h * outputBBoxDims.w;

      float *outputY1 = outputX1 + gridSize;
      float *outputX2 = outputY1 + gridSize;
      float *outputY2 = outputX2 + gridSize;

      for (unsigned int h = 0; h < outputCoverageDims.h; h++)
      {
          for (unsigned int w = 0; w < outputCoverageDims.w; w++)
          {
              int i = w + h * outputCoverageDims.w;
              float confidence = outputCoverageBuffer[classIndex * gridSize + i];

              if (confidence < detectionParams.perClassPreclusterThreshold[classIndex])
                  continue;

              float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

              rectX1Float =
                  outputX1[w + h * outputCoverageDims.w] - gcCenters0[w];
              rectY1Float =
                  outputY1[w + h * outputCoverageDims.w] - gcCenters1[h];
              rectX2Float =
                  outputX2[w + h * outputCoverageDims.w] + gcCenters0[w];
              rectY2Float =
                  outputY2[w + h * outputCoverageDims.w] + gcCenters1[h];

              rectX1Float *= -bboxNorm[0];
              rectY1Float *= -bboxNorm[1];
              rectX2Float *= bboxNorm[0];
              rectY2Float *= bboxNorm[1];

              if (rectX1Float >= (int)networkInfo.width)
                  rectX1Float = networkInfo.width - 1;
              if (rectX2Float >= (int)networkInfo.width)
                  rectX2Float = networkInfo.width - 1;
              if (rectY1Float >= (int)networkInfo.height)
                  rectY1Float = networkInfo.height - 1;
              if (rectY2Float >= (int)networkInfo.height)
                  rectY2Float = networkInfo.height - 1;

              if (rectX1Float < 0)
                  rectX1Float = 0;
              if (rectX2Float < 0)
                  rectX2Float = 0;
              if (rectY1Float < 0)
                  rectY1Float = 0;
              if (rectY2Float < 0)
                  rectY2Float = 0;

              //Prevent underflows
              if(((rectX2Float - rectX1Float) < 0) || ((rectY2Float - rectY1Float) < 0))
                  continue;

              objectList.push_back({ classIndex, rectX1Float,
                       rectY1Float, (rectX2Float - rectX1Float),
                       (rectY2Float - rectY1Float), confidence});
          }
      }
  }
  return true;
}

void ConvertToImgCoordinate(int img_w, int img_h, int net_w, int net_h,
                            NvDsInferDetectionOutput& output) {
  float x_scale = img_w/(float)net_w;
  float y_scale = img_h/(float)net_h;
  for (uint32_t i = 0; i < output.numObjects; i ++) {
    auto& obj = output.objects[i];
    obj.left *= x_scale;
    obj.top *= y_scale;
    obj.width *= x_scale;
    obj.height *= y_scale;
  }
}

int main(int argc, char *argv[]) {
  // args
  argparse::ArgumentParser program("test");
  program.add_argument("--model")
          .required()
          .help("model file");
  program.add_argument("--img")
          .required()
          .help("input image file");
  program.add_argument("--iterators")
          .default_value(1)
          .help("number of runs")
          .scan<'i', int>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return -1;
  }
  std::string model_file = program.get<std::string>("model");
  std::string img_file = program.get<std::string>("img");
  int iterators = program.get<int>("iterators");
  printf("model: %s\n"
         "img: %s\n"
         "iterators: %d\n",
         model_file.data(), img_file.data(), iterators);

  // load engine
  std::ifstream engine_file(model_file, std::ios::binary);
  if (!engine_file.good()) {
    AppError("open engine file failed, %s", model_file.data());
    return -1;
  }
  engine_file.seekg(0, std::ifstream::end);
  int64_t fsize = engine_file.tellg();
  engine_file.seekg(0, std::ifstream::beg);
  std::vector<uint8_t> engine_blob(fsize);
  engine_file.read(reinterpret_cast<char*>(engine_blob.data()), fsize);

  // deserialize
  std::unique_ptr<ICudaEngine> m_engine = nullptr;
  auto runtime = createInferRuntime(sample::gLogger.getTRTLogger());
  m_engine.reset(runtime->deserializeCudaEngine(engine_blob.data(),
                                                engine_blob.size()));
  if (m_engine == nullptr) {
    AppError("deserialize engine failed, %s", model_file.data());
    return -1;
  }
  printf("load engine success, %s\n", __filename(model_file.data()));
  printf("name:%s, bindings:%d, max_batch_size:%d, layers:%d\n", m_engine->getName(),
          m_engine->getNbIOTensors(), m_engine->getMaxBatchSize(), m_engine->getNbLayers());

  // mem alloc
  int net_w = -1, net_h = -1;
  std::vector<void*> bindings ;
  int binding_tensor_size[m_engine->getNbIOTensors()];
  std::vector<NvDsInferLayerInfo> outputLayersInfo;
  for (int i = 0; i < m_engine->getNbIOTensors(); i ++) {
    auto const& name = m_engine->getIOTensorName(i);
    auto const& io_mode = m_engine->getTensorIOMode(name);
    Dims const dims = m_engine->getTensorShape(name);
    auto datatype = m_engine->getTensorDataType(name);
    int tensor_size = samplesCommon::getElementSize(datatype);
    NvDsInferLayerInfo layer;
    layer.inferDims.numDims = dims.nbDims;
    layer.inferDims.numElements = 1;
    printf("bindings[%d], name:%s, io_mode:%d, nbDims:%d, "
           "element_size:%d\n", i, name, (int)io_mode,
           dims.nbDims, samplesCommon::getElementSize(datatype));
    printf("shape: [ ");
    for (int j = 0; j < dims.nbDims; j ++) {
      printf("%d ", dims.d[j]);
      tensor_size *= dims.d[j];
      layer.inferDims.d[j] = dims.d[j];
      layer.inferDims.numElements *= dims.d[j];
    }
    printf("]\n");
    if (io_mode == TensorIOMode::kINPUT) {
      net_h = dims.d[1];
      net_w = dims.d[2];
    }
    else {
      layer.dataType = static_cast<NvDsInferDataType>(datatype);
      layer.bindingIndex = i;
      layer.layerName = name;
      layer.isInput = 0;
      outputLayersInfo.push_back(layer);
    }
    void* ptr;
    cudaMalloc(&ptr, tensor_size);
    binding_tensor_size[i] = tensor_size;
    bindings.push_back(ptr);
  }
  void* out_ptr[2];
  cudaMallocHost(&out_ptr[0], binding_tensor_size[1]);
  cudaMallocHost(&out_ptr[1], binding_tensor_size[2]);

  // create context
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
  if (!context) {
    AppError("create exec context failed, %s", model_file.data());
    return false;
  }

  // preprocess
  cv::Mat img = imread(img_file, cv::IMREAD_COLOR);
  if (img.empty()) {
    AppError("read %s failed", img_file.data());
    return -1;
  }
  cv::Mat blob;
  float scale = 1/255.0;
  cv::Scalar mean = cv::Scalar();
  bool rgb = true;
  bool crop = false;
  cv::dnn::blobFromImage(img, blob, scale, cv::Size(net_w, net_h), mean, rgb, crop);

  // inference
  struct timeval t1, t2;
  int n = iterators;
  gettimeofday(&t1, NULL);
  while (n--) {
    cudaMemcpy(bindings[0], blob.data, binding_tensor_size[0], cudaMemcpyHostToDevice);
    bool status = context->execute(1, bindings.data());
    if (!status) {
      AppError("infer exec failed, %s", model_file.data());
      return -1;
    }
    cudaMemcpy(out_ptr[0], bindings[1], binding_tensor_size[1], cudaMemcpyDeviceToHost);
    cudaMemcpy(out_ptr[1], bindings[2], binding_tensor_size[2], cudaMemcpyDeviceToHost);
  }
  gettimeofday(&t2, NULL);
  float cost_ms = (t2.tv_sec - t1.tv_sec)*1000 + (t2.tv_usec - t1.tv_usec)/1000;
  float fps = 1000*iterators/cost_ms;
  printf("iterators: %d, cost: %fms, inference fps : %f\n", iterators, cost_ms, fps);

  // postprocess
  NvDsInferNetworkInfo networkInfo;
  NvDsInferParseDetectionParams detectionParams;
  std::vector<NvDsInferObjectDetectionInfo> m_ObjectList;
  std::vector<std::vector<NvDsInferObjectDetectionInfo>> m_PerClassObjectList;
  std::vector<NvDsInferDetectionParams> m_PerClassDetectionParams;
  m_ObjectList.clear();
  for (auto & list:m_PerClassObjectList)
    list.clear();
  outputLayersInfo[0].buffer = out_ptr[0];
  outputLayersInfo[1].buffer = out_ptr[1];
  networkInfo.width = net_w;
  networkInfo.height = net_h;
  m_PerClassObjectList.resize(m_NumDetectedClasses);
  detectionParams.numClassesConfigured = m_NumDetectedClasses;
  detectionParams.perClassPreclusterThreshold.resize(m_NumDetectedClasses);
  detectionParams.perClassPreclusterThreshold[0] = 0.4;
  detectionParams.perClassPreclusterThreshold[1] = 0.2;
  detectionParams.perClassPreclusterThreshold[2] = 0.2;
  detectionParams.perClassPreclusterThreshold[3] = 0.2;
  m_PerClassDetectionParams.resize(m_NumDetectedClasses);
  for (auto& det_params : m_PerClassDetectionParams) {
    det_params.nmsIOUThreshold = 0.5;
    det_params.postClusterThreshold = 0.0;
    det_params.topK = 20;
  }
  m_Labels.resize(m_NumDetectedClasses);
  m_Labels[0].push_back("Car");
  m_Labels[1].push_back("Bicycle");
  m_Labels[2].push_back("Person");
  m_Labels[3].push_back("Roadsign");
  // parse bbox
  parseBoundingBox(outputLayersInfo, networkInfo, detectionParams, m_ObjectList);
  preClusteringThreshold(detectionParams, m_ObjectList);
  for (auto & object:m_ObjectList) {
    m_PerClassObjectList[object.classId].emplace_back(object);
  }
  NvDsInferDetectionOutput output = {0};
  // nms
  clusterAndFillDetectionOutputNMS(m_PerClassObjectList, m_PerClassDetectionParams, output);
  // convert to orig image coordinate
  ConvertToImgCoordinate(img.cols, img.rows, net_w, net_h, output);

  // result
  printf("result:\n");
  for (uint32_t i = 0; i < output.numObjects; i ++) {
    auto& obj = output.objects[i];
    printf("output[%d],left:%f,top:%f,width:%f,height:%f,classIndex:%d,label:%s,confidence:%f\n",
        i, obj.left, obj.top, obj.width, obj.height, obj.classIndex, obj.label, obj.confidence);
    cv::Rect rect_(obj.left, obj.top, obj.width, obj.height);
    cv::rectangle(img, rect_, cv::Scalar(0,255,0), 2);
  }
  cv::imwrite("result.jpg", img);

  // release
  for (int i = 0; i < m_engine->getNbIOTensors(); i ++) {
    cudaFree(bindings[i]);
  }
  cudaFreeHost(out_ptr[0]);
  cudaFreeHost(out_ptr[1]);

  return 0;
}

