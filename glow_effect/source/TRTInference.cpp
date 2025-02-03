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

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Measure Segmentation Inference (Single Image)
 ////////////////////////////////////////////////////////////////////////////////////////////////////

void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {
	std::cout << "STARTING measure_trt_performance" << std::endl;

	// Create custom logger and load the TensorRT plan file.
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	// Deserialize the CUDA engine and create an execution context.
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate pinned host memory for input.
	float* h_input;
	int input_size = img_tensor.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	// Retrieve binding information.
	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	// Collect output binding indices and names.
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	// Set input dimensions (assuming 4D: NCHW).
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	// Prepare vectors to store output pointers and binding handles.
	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	// Create a CUDA stream for asynchronous operations.
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for the input.
	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	// Copy input tensor data to pinned host memory.
	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

	// Asynchronously copy input data from host to device.
	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// Allocate memory for all output bindings.
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);
		// Handle dynamic dimensions by copying corresponding values from input.
		for (int j = 0; j < outputDims.nbDims; ++j) {
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

	// Prepare CUDA events for latency measurement.
	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm-up runs.
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	// Measure inference latency.
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

	// Copy the last output back to host.
	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Compute and print basic statistics.
	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;

	// Clean up CUDA events.
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Wrap the output in a Torch tensor.
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

	// Compute argmax across the classes dimension.
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);

	// Simple scaling for grayscale visualization.
	int scale = 255 / 21;
	auto image_post = class_labels * scale;

	cout << "\nimage_post dimensions: ";
	for (int i = 0; i < image_post.dim(); ++i) {
		cout << image_post.size(i) << " ";
	}
	cout << std::endl;

	// Convert from NCHW to HWC.
	auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);
	cout << "permuted_img dimensions: ";
	for (int i = 0; i < permuted_img.dim(); ++i) {
		cout << permuted_img.size(i) << " ";
	}
	cout << std::endl;

	// Convert to OpenCV Mat.
	cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());
	cout << "Segmentation visualization ready." << endl;

	// Clean up allocated memory.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Measure Segmentation Inference (Batch)
////////////////////////////////////////////////////////////////////////////////////////////////////

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

	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
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
		outputDims = engine->getBindingDimensions(i);
		for (int j = 0; j < outputDims.nbDims; ++j) {
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

	// Convert each item in the batch to an OpenCV grayscale Mat.
	for (int i = 0; i < batch; ++i) {
		auto single_image_post = image_post[i].squeeze().to(torch::kU8);
		cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());
		grayscale_images.push_back(cv_img.clone());
	}

	// Cleanup.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Measure Super-Resolution Inference
////////////////////////////////////////////////////////////////////////////////////////////////////

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

	float* h_input;
	int input_size = img_tensor.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
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

	// Handle output bindings. For super-resolution, output dimensions may be adjusted.
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);
		for (int j = 0; j < outputDims.nbDims; ++j) {
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}
		// Example: modify output height and width (model-specific adjustment).
		outputDims.d[2] *= 2;
		outputDims.d[3] *= 4;

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

	// Convert final output to a single-channel OpenCV Mat.
	cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);
	cv::Mat clipped_image_data;
	cv::min(image_data, 1.0, clipped_image_data);
	cv::max(clipped_image_data, 0.0, clipped_image_data);
	clipped_image_data *= 255;
	clipped_image_data.convertTo(clipped_image_data, CV_8U);

	// Cleanup.
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
