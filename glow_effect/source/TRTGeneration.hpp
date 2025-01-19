#ifndef TRT_GENERATION_HPP
#define TRT_GENERATION_HPP
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/quality.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
#include "ImageProcessingUtil.hpp" 


struct DynamicShapeRange { nvinfer1::Dims minDims; nvinfer1::Dims optDims; nvinfer1::Dims maxDims; };

class TRTGeneration {
public:
    // Generates a TRT plan from an ONNX model
    static vector<int> gen_trt_plan(const string& onnxModelPath, const string& planFilePath,
                                         const string& precision_mode, bool dynamic,
                                         DynamicShapeRange& dynamicShapeRange);


    class CustomLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            cout << msg << endl;
        }
    };


    class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        Int8EntropyCalibrator(int batchSize, int channelNum, const vector<string>& image_paths, const string& cache_file, const vector<string>& input_tensor_names)
            : batchSize(batchSize), channelNum(channelNum), imagePaths(image_paths), cacheFile(cache_file), inputTensorNames(input_tensor_names), currentBatchIndex(0)  {
            const int inputSize = batchSize * channelNum * 1024 * 2048 * sizeof(float);
            
            if (cudaMalloc(&deviceInput, inputSize) != cudaSuccess) {
                throw runtime_error("cudaMalloc failed!");
            }
        }

        int getBatchSize() const noexcept override {
            return batchSize;
        }

        bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
            
            cout << "getBatch called with " << nbBindings << " bindings." << endl;
            cout << "getBatch - imagePaths address: " << &imagePaths << endl;
            
            if (currentBatchIndex >= imagePaths.size()) {
                cout << "No more batches to load" << endl;
                cout << "currentBatchIndex: " << currentBatchIndex << endl;
                cout << "END imagePaths.size(): " << imagePaths.size() << endl;
                return false;
            }
            for (int i = 0; i < nbBindings; ++i) {
                bool nameFound = false;
                for (const auto& tensorName : inputTensorNames) {
                    if (strcmp(names[i], tensorName.c_str()) == 0) {
                        bindings[i] = deviceInput;
                        nameFound = true;
                        break;
                    }
                }
                if (!nameFound) {
                    cout << "FALSE 3 - Binding name not found: " << names[i] << endl;
                    return false;
                }
            }
            loadBatch();
            cout << "Batch " << currentBatchIndex / batchSize << " loaded successfully." << endl;
            return true;
        }

        const void* readCalibrationCache(size_t& length) noexcept override {
            length = calibrationCache.size();
            return length ? calibrationCache.data() : nullptr;
        }

        void writeCalibrationCache(const void* cache, size_t length) noexcept override {
            calibrationCache.assign(static_cast<const char*>(cache), static_cast<const char*>(cache) + length);
        }

        ~Int8EntropyCalibrator() noexcept {
            cudaFree(deviceInput);
        }

    private:
        int batchSize;
        int channelNum;
        vector<string> imagePaths;
        string cacheFile;
        const vector<string> inputTensorNames;
        void* deviceInput; // Pointer to GPU memory for input data
        vector<char> calibrationCache; // For storing calibration cache
        size_t currentBatchIndex; // Index of the current batch

       void loadBatch() {
            std::vector<torch::Tensor> img_tensors;
            img_tensors.reserve(batchSize);

            // Load a batch of images
            for (int i = 0; i < batchSize; ++i) {
                size_t imageIndex = currentBatchIndex + i;
                if (imageIndex < imagePaths.size()) {
                    torch::Tensor img_tensor = ImageProcessingUtil::process_img(imagePaths[imageIndex]);
                    img_tensor = img_tensor.contiguous().to(torch::kCPU);
                    
                    // Ensure the image tensor has the correct shape [C, H, W]
                    if (img_tensor.dim() == 4) {
                        img_tensor = img_tensor.squeeze(0);
                    }
                    
                    img_tensors.push_back(img_tensor);
                } else {
                    // If we have fewer images than the batch size, pad with zeros
                    torch::Tensor padding_tensor = torch::zeros({channelNum, 1024, 2048}, torch::kFloat32);
                    img_tensors.push_back(padding_tensor);
                }
            }
            std::cout << "img_tensors size: " << img_tensors.size() << std::endl;

            // Create a TensorList from the vector of tensors
            torch::TensorList tensor_list(img_tensors);

            // Concatenate the image tensors along the batch dimension
            torch::Tensor batch_tensor = torch::stack(tensor_list, 0);
            std::cout << "Batch tensor dimensions: " << batch_tensor.sizes() << std::endl;

            // Copy the batch tensor data to the host buffer
            float* hostDataBuffer = new float[batchSize * channelNum * 1024 * 2048];
            std::memcpy(hostDataBuffer, batch_tensor.data_ptr<float>(), batchSize * channelNum * 1024 * 2048 * sizeof(float));

            // Copy the batch data from the host buffer to the device buffer
            if (cudaMemcpy(deviceInput, hostDataBuffer, batchSize * channelNum * 1024 * 2048 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                delete[] hostDataBuffer;
                throw std::runtime_error("cudaMemcpy failed!");
            }

            delete[] hostDataBuffer;
            currentBatchIndex += batchSize; // Move to the next batch
        }
    };
};

#endif