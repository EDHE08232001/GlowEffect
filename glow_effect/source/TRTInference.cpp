/**
 * @file TRTInference.cpp
 * @brief Implementation of TensorRT inference routines for segmentation and super-resolution.
 *
 * This file provides functions to measure inference performance, execute batched or single-image
 * segmentation, and process super-resolution outputs.
 */

#include "TRTInference.hpp"
#include "ImageProcessingUtil.hpp"
#include "nvToolsExt.h"
#include "helper_cuda.h"  // For checkCudaErrors
#include <future>
#include <thread>
#include <mutex>
#include <iterator>

 //--------------------------------------------------------------------------
 // Measure Segmentation Inference (Single Image)
 //--------------------------------------------------------------------------
void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {
	std::cout << "STARTING measure_trt_performance" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Use Dims4 for input and output.
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	int input_size = img_tensor.numel();
	float* h_input;
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	// Collect output binding indices.
	int numBindings = engine->getNbBindings();
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	// Set input dimensions (NCHW).
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));
	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// For each output binding, copy dimensions manually.
	for (int i : outputBindingIndices) {
		// Get the raw dims
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0) {  // Handle dynamic dimensions.
				outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm-up runs.
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		char str_buf[100];
		std::sprintf(str_buf, "frame%03d", i);
		nvtxRangePushA(str_buf);
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
		nvtxRangePop();
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	cout << "\nLast output tensor dimensions: ";
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		cout << last_output_tensor.size(i) << " ";
	}
	cout << std::endl;

	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);
	int scale = 255 / 21;
	auto image_post = class_labels * scale;

	cout << "\nimage_post dimensions: ";
	for (int i = 0; i < image_post.dim(); ++i) {
		cout << image_post.size(i) << " ";
	}
	cout << std::endl;

	auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);
	cout << "permuted_img dimensions: ";
	for (int i = 0; i < permuted_img.dim(); ++i) {
		cout << permuted_img.size(i) << " ";
	}
	cout << std::endl;

	cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());
	cout << "Segmentation visualization ready." << endl;

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}

	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);
}

//--------------------------------------------------------------------------
// Measure Segmentation Inference (Batch) - Original Version
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul(const string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
	std::vector<cv::Mat> grayscale_images;
	std::cout << "STARTING measure_segmentation_trt_performance_mul" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	float* h_input;
	int input_size = img_tensor_batch.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < engine->getNbBindings(); ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	inputDims.d[0] = img_tensor_batch.size(0);
	inputDims.d[1] = img_tensor_batch.size(1);
	inputDims.d[2] = img_tensor_batch.size(2);
	inputDims.d[3] = img_tensor_batch.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor_batch.data_ptr<float>(), input_size * sizeof(float));

	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	for (int i : outputBindingIndices) {
		// Copy dimensions manually from raw dims.
		nvinfer1::Dims dims = context->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0)
				outputDims.d[j] = inputDims.d[j];
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		char str_buf[100];
		std::sprintf(str_buf, "frame%03d", i);
		nvtxRangePushA(str_buf);
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
		nvtxRangePop();
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	cout << "\nLast output tensor dimensions: ";
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		cout << last_output_tensor.size(i) << " ";
	}
	cout << std::endl;

	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);
	int scale = 255 / 21;
	auto image_post = class_labels * scale;

	for (int i = 0; i < batch; ++i) {
		auto single_image_post = image_post[i].squeeze().to(torch::kU8);
		cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());
		grayscale_images.push_back(cv_img.clone());
	}

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);

	return grayscale_images;
}

//--------------------------------------------------------------------------
// New Function: Measure Segmentation Inference (Batch) Concurrent Version
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent(
	const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {

	std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent (multi-stream concurrent version)" << std::endl;

	// Create runtime and deserialize the engine.
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!engine) {
		std::cerr << "Failed to deserialize engine in concurrent segmentation." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Determine the total number of images in the batch.
	int totalBatch = img_tensor_batch.size(0);
	int numThreads = 2;
	int subBatch = (totalBatch + numThreads - 1) / numThreads;

	// allResults will hold the segmentation output for each image.
	std::vector<cv::Mat> allResults(totalBatch);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	for (int t = 0; t < numThreads; ++t) {
		threads.emplace_back([&, t]() {
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Failed to create execution context for thread " << t << std::endl;
				return;
			}
			cudaStream_t stream;
			checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

			int startIdx = t * subBatch;
			int endIdx = std::min(startIdx + subBatch, totalBatch);
			int validCount = endIdx - startIdx; // Number of valid images in this sub-batch.
			if (validCount <= 0)
				return;

			// Slice the global tensor to get the sub-batch.
			torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);
			// Remove any extra singleton dimension (e.g. [N,1,3,384,384] -> [N,3,384,384]).
			if ((subTensor.dim() == 5 && subTensor.size(1) == 1) ||
				(subTensor.dim() == 4 && subTensor.size(1) == 1))
			{
				subTensor = subTensor.squeeze(1);
			}
			// Pad the sub-batch to have exactly 4 images if needed.
			if (subTensor.size(0) < 4) {
				int pad = 4 - subTensor.size(0);
				torch::Tensor lastFrame = subTensor[subTensor.size(0) - 1].unsqueeze(0);
				torch::Tensor padTensor = lastFrame.repeat({ pad, 1, 1, 1 });
				subTensor = torch::cat({ subTensor, padTensor }, 0);
			}

			int inputSize = subTensor.numel();
			float* h_input = nullptr;
			checkCudaErrors(cudaMallocHost((void**)&h_input, inputSize * sizeof(float)));
			std::memcpy(h_input, subTensor.data_ptr<float>(), inputSize * sizeof(float));

			void* d_input = nullptr;
			checkCudaErrors(cudaMalloc(&d_input, inputSize * sizeof(float)));
			checkCudaErrors(cudaMemcpyAsync(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));

			nvinfer1::Dims4 inputDims;
			inputDims.d[0] = subTensor.size(0);
			inputDims.d[1] = subTensor.size(1);
			inputDims.d[2] = subTensor.size(2);
			inputDims.d[3] = subTensor.size(3);
			context->setBindingDimensions(0, inputDims);

			// Prepare output buffers.
			std::vector<void*> bindings;
			bindings.push_back(d_input);
			int numBindings = engine->getNbBindings();
			std::vector<float*> h_outputs;
			std::vector<void*> d_outputs;
			nvinfer1::Dims4 outputDims;
			for (int i = 1; i < numBindings; ++i) {
				nvinfer1::Dims dims = context->getBindingDimensions(i);
				outputDims.nbDims = dims.nbDims;
				for (int j = 0; j < dims.nbDims; ++j) {
					outputDims.d[j] = dims.d[j];
					if (outputDims.d[j] < 0)
						outputDims.d[j] = inputDims.d[j];
				}
				int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
				float* h_output = nullptr;
				checkCudaErrors(cudaMallocHost((void**)&h_output, outputSize * sizeof(float)));
				void* d_output = nullptr;
				checkCudaErrors(cudaMalloc(&d_output, outputSize * sizeof(float)));
				h_outputs.push_back(h_output);
				d_outputs.push_back(d_output);
				bindings.push_back(d_output);
			}

			// Optional warm-up runs.
			for (int i = 0; i < 3; ++i) {
				context->enqueueV2(bindings.data(), stream, nullptr);
			}

			if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
				std::cerr << "TensorRT enqueueV2 failed in thread " << t << std::endl;
				exit(EXIT_FAILURE);
			}

			int outSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
			float* lastOutput = h_outputs.back();
			checkCudaErrors(cudaMemcpyAsync(lastOutput, d_outputs.back(), outSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
			cudaStreamSynchronize(stream);

			auto outputTensor = torch::from_blob(lastOutput, { outputDims.d[0], outputDims.d[1], outputDims.d[2], outputDims.d[3] }, torch::kFloat32);
			auto maxOut = torch::max(outputTensor, 1);
			auto segMask = std::get<1>(maxOut); // Expected shape: [4, H, W]
			int scaleFactor = 255 / 21;
			auto imagePost = segMask * scaleFactor;

			// Only use the first validCount outputs (the rest are padded).
			std::vector<cv::Mat> localResults;
			for (int i = 0; i < validCount; ++i) {
				auto single = imagePost[i].to(torch::kU8);
				cv::Mat result(single.size(0), single.size(1), CV_8UC1, single.data_ptr<uchar>());
				localResults.push_back(result.clone());
			}

			{
				std::lock_guard<std::mutex> lock(resultMutex);
				for (int i = 0; i < validCount; ++i) {
					allResults[startIdx + i] = localResults[i];
				}
			}

			// Free allocated memory and resources.
			cudaFreeHost(h_input);
			cudaFree(d_input);
			for (auto ptr : h_outputs) {
				cudaFreeHost(ptr);
			}
			for (auto dptr : d_outputs) {
				cudaFree(dptr);
			}
			cudaStreamDestroy(stream);
			context->destroy();
			});
	}

	for (auto& t : threads)
		t.join();

	engine->destroy();
	runtime->destroy();

	return allResults;
}


//--------------------------------------------------------------------------
// Measure Super-Resolution Inference
//--------------------------------------------------------------------------
void TRTInference::measure_trt_performance(const string& trt_plan,
	const string& original_image_path,
	torch::Tensor img_tensor,
	int num_trials,
	bool compare_img_bool) {

	std::cout << "STARTING measure_trt_performance" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	int input_size = img_tensor.numel();
	float* h_input;
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < engine->getNbBindings(); ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));
	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	for (int i : outputBindingIndices) {
		nvinfer1::Dims dims = context->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);
	cv::Mat clipped_image_data;
	cv::min(image_data, 1.0, clipped_image_data);
	cv::max(clipped_image_data, 0.0, clipped_image_data);
	clipped_image_data *= 255;
	clipped_image_data.convertTo(clipped_image_data, CV_8U);

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);
}
