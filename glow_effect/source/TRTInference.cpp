/**
 * @file TRTInference.cpp
 * @brief Implementation of TensorRT inference routines for segmentation and super-resolution.
 */

 /**
  * @note You probably need to change the output path labeled with "pngOutput"
  *       to match your own directory structure.
  */

#include "TRTInference.hpp"
#include "ImageProcessingUtil.hpp" 
#include "nvToolsExt.h"

  /**
   * @brief Perform TensorRT segmentation inference on a single image and measure performance.
   *
   * This function:
   * 1. Loads a TensorRT plan file into memory.
   * 2. Deserializes the engine and creates an execution context.
   * 3. Allocates pinned (host) and device memory for input/output.
   * 4. Copies input data to device memory.
   * 5. Runs warm-up inference to ensure GPU caches and contexts are prepared.
   * 6. Measures the latency over a specified number of trials.
   * 7. Copies the last output tensor back to host, calculates min/max/avg, and saves a visualization.
   *
   * @param[in] trt_plan   Path to the serialized TensorRT engine plan file.
   * @param[in] img_tensor A 4D tensor (NCHW) containing the preprocessed input image data.
   * @param[in] num_trials Number of inference runs for performance measurement.
   */
void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {

	std::cout << "STARTING measure_trt_performance" << std::endl;

	// Create custom logger for TensorRT and load the serialized plan file
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	// Read the plan file in binary mode
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	// Deserialize the CUDA engine from the plan
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate pinned host memory for the input tensor
	float* h_input;
	int input_size = img_tensor.numel();  // total number of elements in the input tensor
	cudaMallocHost((void**)&h_input, input_size * sizeof(float)); // pinned memory

	// Retrieve the number of bindings from the engine
	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	// We collect the indices of all output bindings
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		std::string outputTensorName = engine->getBindingName(i);
		outputTensorNames.push_back(outputTensorName);
	}

	// Extract the dimensions from the input tensor (assuming the format is 4D: NCHW)
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);

	// Set the binding dimensions for the input (binding index 0)
	context->setBindingDimensions(0, inputDims);

	// We'll store output pointers (host and device) in vectors
	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	// Create a CUDA stream for asynchronous operations
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for the input and add it to the bindings list
	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	// Copy data from the img_tensor to the pinned host memory
	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

	// Asynchronously copy input data from host to device
	cudaError_t mallocErr = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (mallocErr != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(mallocErr) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// Handle dynamic dimensions for all output bindings
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);

		// If the model uses dynamic dimensions, fill in the batch/width/height from inputDims
		for (int j = 0; j < outputDims.nbDims; ++j) {
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}

		// Compute total output size
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];

		// Allocate host and device memory for this output
		float* h_output = new float[outputSize];
		void* d_output;
		cudaError_t status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}

		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	// Prepare to measure latency
	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm-up (run inference several times before actual measurements)
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	// Record the start event and run multiple inference trials
	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		// Provide NVTX annotation (optional), helpful for profiling in Nsight Systems
		char str_buf[100];
		std::sprintf(str_buf, "frame%03d", i);
		nvtxRangePushA(str_buf);

		// Run asynchronous inference
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
		nvtxRangePop();
	}
	// Record the stop event and synchronize
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	// Calculate elapsed time in milliseconds
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	// Copy the last output tensor back to host (the last item in h_outputs / d_outputs)
	float* last_h_output = h_outputs.back();
	void* last_d_output = d_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Compute min, max, and avg values in the final output (mostly for debugging/analysis)
	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	// Compute average latency over the measured trials
	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;

	// Clean up CUDA events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Prepare the output for saving or further post-processing:
	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];

	// Wrap the output in a Torch tensor for easy post-processing
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	// Print shape for debugging
	std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		std::cout << last_output_tensor.size(i) << " ";
	}
	std::cout << std::endl;

	// Compute the argmax across the classes dimension
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);

	// Simple scaling to produce a grayscale visualization
	int scale = 255 / 21;
	auto scale_tensor = torch::tensor(scale, class_labels.options());
	auto image_post = class_labels * scale;

	std::cout << "\nNumber of dimensions(image_post): " << image_post.dim() << std::endl;
	for (int i = 0; i < image_post.dim(); ++i) {
		std::cout << image_post.size(i) << " ";
	}
	std::cout << std::endl;

	// Convert from NCHW to HWC (though for single-channel, it's effectively HxWx1)
	auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);
	std::cout << "Number of dimensions(permuted_img): " << permuted_img.dim() << std::endl;
	for (int i = 0; i < permuted_img.dim(); ++i) {
		std::cout << permuted_img.size(i) << " ";
	}

	// Convert to OpenCV Mat
	cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());

	// Save the output
	try {
		cv::imwrite("pngOutput/trt_seg_output_scaled.png", cv_img); // Replace with your own path
		cout << "Saved IMG: trt_seg_output_scaled" << endl;
	}
	catch (const cv::Exception& ex) {
		cerr << "Failed to save image trt_seg_output_scaled because ERROR:" << ex.what() << endl;
	}

	// Clean up memory
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



/**
 * @brief Perform TensorRT segmentation inference on multiple images at once (batch) and measure performance.
 *
 * This function:
 * 1. Loads a TensorRT plan file and deserializes it into a CUDA engine.
 * 2. Handles batched input (4D NCHW).
 * 3. Allocates memory for input/output on host (pinned) and device.
 * 4. Executes inference multiple times for latency measurement.
 * 5. Retrieves the final output, converts each batch slice into a grayscale visualization.
 * 6. Returns a vector of OpenCV Mats for further usage (e.g., post-processing or saving).
 *
 * @param[in] trt_plan         Path to the TensorRT engine plan file.
 * @param[in] img_tensor_batch A 4D tensor (NCHW) containing batched preprocessed images.
 * @param[in] num_trials       Number of inference runs for performance measurement.
 * @return A vector of OpenCV Mats, each representing a grayscale segmentation map for the corresponding batch entry.
 */
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul(const string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
	std::vector<cv::Mat> grayscale_images;  // will store each grayscale image

	std::cout << "STARTING measure_segmentation_trt_performance_mul" << std::endl;

	// Create logger and runtime
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	// Read plan file
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	// Deserialize engine
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate pinned host memory for batched input
	float* h_input;
	int input_size = img_tensor_batch.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float)); // pinned memory

	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	// Identify output bindings
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		std::string outputTensorName = engine->getBindingName(i);
		outputTensorNames.push_back(outputTensorName);
	}

	// Set input dimensions (assuming NCHW format)
	inputDims.d[0] = img_tensor_batch.size(0);
	inputDims.d[1] = img_tensor_batch.size(1);
	inputDims.d[2] = img_tensor_batch.size(2);
	inputDims.d[3] = img_tensor_batch.size(3);
	context->setBindingDimensions(0, inputDims);

	// Prepare vectors to hold output data (host and device)
	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	// Create CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for input
	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	// Copy data from tensor to pinned host buffer
	std::memcpy(h_input, img_tensor_batch.data_ptr<float>(), input_size * sizeof(float));

	// Transfer data to GPU
	cudaError_t mallocErr = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (mallocErr != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(mallocErr) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// Allocate and bind outputs
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);
		// Handle dynamic shape
		for (int j = 0; j < outputDims.nbDims; ++j) {
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}

		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		cudaError_t status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}

		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	// Performance measurement setup
	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm up inference
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	// Time multiple runs of inference
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

	// Copy the last output to host
	float* last_h_output = h_outputs.back();
	void* last_d_output = d_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Print stats for the last output
	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	// Calculate average latency
	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;

	// Clean up events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Convert the last output into a Torch tensor for easy post-processing
	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	// Debug shape printout
	std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		std::cout << last_output_tensor.size(i) << " ";
	}
	std::cout << std::endl;

	// Find the class index with the highest value (argmax)
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);

	// Scale for visualization
	int scale = 255 / 21;
	auto scale_tensor = torch::tensor(scale, class_labels.options());
	auto image_post = class_labels * scale;

	// Convert each item in the batch to an OpenCV grayscale Mat
	for (int i = 0; i < batch; ++i) {
		// Extract i-th image, convert to 8-bit
		auto single_image_post = image_post[i].squeeze().to(torch::kU8);
		cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());

		// Store the grayscale image
		grayscale_images.push_back(cv_img.clone());
		try {
			// Example of saving each image to disk
			cv::imwrite("pngOutput/trt_seg_output_scaled_" + std::to_string(i) + ".png", cv_img);
			cout << "Saved IMG: trt_seg_output_scaled_" + std::to_string(i) << endl;
		}
		catch (const cv::Exception& ex) {
			cerr << "Failed to save image trt_seg_output_scaled_" + std::to_string(i) + " because ERROR:" << ex.what() << endl;
		}
	}

	// Cleanup
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

	// Return the grayscale segmentation maps
	return grayscale_images;
}



/**
 * @brief Perform TensorRT inference for super-resolution and measure performance.
 *
 * This function:
 * 1. Deserializes a TensorRT plan file for super-resolution.
 * 2. Allocates pinned host memory and device memory for input/output.
 * 3. Runs multiple warm-up inferences.
 * 4. Measures latency for a specified number of trials.
 * 5. Copies the final output back to host, clips values to [0,1], multiplies by 255, and converts to 8-bit.
 * 6. Optionally compares the output to a provided original image if compare_img_bool is true.
 *
 * @param[in] trt_plan           Path to the TensorRT plan file (super-resolution model).
 * @param[in] original_image_path Path to the original image file (for comparison).
 * @param[in] img_tensor         A 4D input tensor (NCHW) for the super-resolution model.
 * @param[in] num_trials         Number of inference runs for latency measurement.
 * @param[in] compare_img_bool   If true, compare the final output with the original image (e.g., using PSNR/SSIM).
 */
void TRTInference::measure_trt_performance(const string& trt_plan,
	const string& original_image_path,
	torch::Tensor img_tensor,
	int num_trials,
	bool compare_img_bool) {

	std::cout << "STARTING measure_trt_performance" << std::endl;

	// Create TRT runtime using custom logger
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	// Read plan file in binary
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	// Deserialize CUDA engine and create context
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate pinned host memory for the input
	float* h_input;
	int input_size = img_tensor.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float)); // pinned memory

	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	// Collect output bindings
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		std::string outputTensorName = engine->getBindingName(i);
		outputTensorNames.push_back(outputTensorName);
	}

	// Set the dimensions based on the input tensor's shape (NCHW)
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	// Vectors to store device pointers for outputs
	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	// Create a CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for input
	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	// Copy data from the torch tensor to pinned memory
	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

	// Async copy from host to device
	cudaError_t mallocErr = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (mallocErr != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(mallocErr) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// Handle outputs (assumes we are upscaling image dimensions for SR)
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);

		// If we had dynamic shape support, fill it in from input
		for (int j = 0; j < outputDims.nbDims; ++j) {
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}

		// IMPORTANT: Manually modify the output dimensions if your SR model expects it.
		// This example scales outputDims height by 2 and width by 4 (arbitrary for demonstration).
		outputDims.d[2] *= 2;
		outputDims.d[3] *= 4;

		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		cudaError_t status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}

		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	// Prepare to measure performance
	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm up
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	// Record time for multiple trials
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

	// Copy the last output back to host
	float* last_h_output = h_outputs.back();
	void* last_d_output = d_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Calculate min, max, avg for debugging
	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	// Average latency
	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Convert the final float output into a single-channel CV_32F Mat
	cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);

	// Clip to [0, 1] for typical image normalization
	cv::Mat clipped_image_data;
	cv::min(image_data, 1.0, clipped_image_data);
	cv::max(clipped_image_data, 0.0, clipped_image_data);

	// Multiply by 255 for 8-bit display
	clipped_image_data *= 255;
	clipped_image_data.convertTo(clipped_image_data, CV_8U);

	// Save the final output
	try {
		cv::imwrite("pngOutput/trt_output.png", clipped_image_data); // custom path
		cout << "Saved IMG: trt_output" << endl;

		// Optionally compare with original
		cv::Mat original_image = cv::imread(original_image_path);
		if (original_image.empty()) {
			cerr << "Error: Original image not found or unable to read." << endl;
		}
		else if (compare_img_bool) {
			// If compare_img_bool is true, run user-defined comparison (e.g., PSNR, SSIM, etc.)
			ImageProcessingUtil::compareImages(clipped_image_data, original_image);
		}

	}
	catch (const cv::Exception& ex) {
		cerr << "Failed to save the image: " << ex.what() << endl;
	}

	// Cleanup
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