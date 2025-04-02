/**
* you probably need to change the output path labeled 
*/

 /* @file TRTInference.cpp
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
#include "segmentation_kernels.h"
#include "TRTInference.hpp"

// <<<<<<< HEAD

 // Add these external variable declarations
extern int param_KeyLevel;  // Defined in control_gui.cpp
extern int param_KeyScale;  // Defined in control_gui.cpp 
extern int default_scale;   // Defined in control_gui.cpp



IRuntime* TRTInference::s_runtime = nullptr;
ICudaEngine* TRTInference::s_engine = nullptr;
IExecutionContext* TRTInference::s_context = nullptr;
bool TRTInference::s_initialized = false;

cudaGraph_t TRTInference::s_graph = nullptr;
cudaGraphExec_t TRTInference::s_graphExec = nullptr;
bool TRTInference::s_graphInitialized = false;
void* TRTInference::s_d_input = nullptr;
std::vector<void*> TRTInference::s_d_outputs;
std::vector<float*> TRTInference::s_h_outputs;
std::vector<void*> TRTInference::s_bindings;
cudaStream_t TRTInference::s_stream = nullptr;
nvinfer1::Dims TRTInference::s_outputDims;

bool TRTInference::initializeTRTEngine(const std::string& trt_plan) {
	// If already initialized, return success
	if (s_initialized) {
		return true;
	}

	std::cout << "Initializing TensorRT engine from plan: " << trt_plan << std::endl;

	// Create logger and runtime
	TRTGeneration::CustomLogger myLogger;
	s_runtime = createInferRuntime(myLogger);

	// Read the TensorRT engine plan file (binary mode)
	std::ifstream planFile(trt_plan, std::ios::binary);
	if (!planFile.is_open()) {
		std::cerr << "Failed to open plan file: " << trt_plan << std::endl;
		return false;
	}

	std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
		std::istreambuf_iterator<char>());

	// Deserialize the engine
	s_engine = s_runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!s_engine) {
		std::cerr << "Failed to deserialize engine." << std::endl;
		return false;
	}

	// Create execution context
	s_context = s_engine->createExecutionContext();
	if (!s_context) {
		std::cerr << "Failed to create execution context." << std::endl;
		s_engine->destroy();
		s_runtime->destroy();
		return false;
	}

	// Create a persistent CUDA stream
	cudaStreamCreate(&s_stream);

	s_initialized = true;
	std::cout << "TensorRT engine initialized successfully." << std::endl;
	return true;
}

bool TRTInference::initializeCudaGraph(const torch::Tensor& sample_tensor) {
	if (!s_initialized) {
		std::cerr << "TensorRT engine must be initialized before CUDA graph." << std::endl;
		return false;
	}

	if (s_graphInitialized) {
		return true; // Graph already initialized
	}

	std::cout << "Initializing CUDA graph for inference..." << std::endl;

	// Calculate input size
	int input_size = sample_tensor.numel();

	// Allocate persistent device memory for input
	cudaError_t status = cudaMalloc(&s_d_input, input_size * sizeof(float));
	if (status != cudaSuccess) {
		std::cerr << "Failed to allocate persistent device memory for input." << std::endl;
		return false;
	}

	// Initialize the bindings vector with input buffer
	s_bindings.clear();
	s_bindings.push_back(s_d_input);

	// Set up output bindings
	int numBindings = s_engine->getNbBindings();
	std::vector<int> outputBindingIndices;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
	}

	// Extract dimensions from the input tensor (assuming shape [1, C, H, W])
	nvinfer1::Dims4 inputDims;
	inputDims.d[0] = sample_tensor.size(0);  // should be 1
	inputDims.d[1] = sample_tensor.size(1);
	inputDims.d[2] = sample_tensor.size(2);
	inputDims.d[3] = sample_tensor.size(3);
	s_context->setBindingDimensions(0, inputDims);

	// Allocate persistent memory for outputs
	s_d_outputs.clear();
	s_h_outputs.clear();

	for (int i : outputBindingIndices) {
		s_outputDims = s_engine->getBindingDimensions(i);
		// Fix dynamic dimensions if necessary
		for (int j = 0; j < s_outputDims.nbDims; ++j) {
			if (s_outputDims.d[j] < 0) {
				s_outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = s_outputDims.d[0] * s_outputDims.d[1] * s_outputDims.d[2] * s_outputDims.d[3];
		float* h_output = new float[outputSize];  // allocate host memory for output
		void* d_output;
		status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "Device memory allocation for output failed" << std::endl;
			// Clean up already allocated resources
			cudaFree(s_d_input);
			for (float* h_output : s_h_outputs) {
				delete[] h_output;
			}
			for (void* d_output : s_d_outputs) {
				cudaFree(d_output);
			}
			return false;
		}
		s_h_outputs.push_back(h_output);
		s_d_outputs.push_back(d_output);
		s_bindings.push_back(d_output);
	}

	// --- Warm-Up Round ---
	// Run one inference call to finish any lazy initialization.
	if (!s_context->enqueueV2(s_bindings.data(), s_stream, nullptr)) {
		std::cerr << "Warm-up inference call failed." << std::endl;
		return false;
	}
	cudaStreamSynchronize(s_stream);
	// --- End Warm-Up Round ---

	// Begin graph capture
	cudaGraphCreate(&s_graph, 0);
	cudaStreamBeginCapture(s_stream, cudaStreamCaptureModeGlobal);

	// Copy sample data to GPU (this is a placeholder operation in the graph)
	cudaMemcpyAsync(
		s_d_input,
		sample_tensor.data_ptr<float>(),
		input_size * sizeof(float),
		cudaMemcpyDeviceToDevice,
		s_stream);

	// Execute inference (this will be recorded in the graph)
	s_context->enqueueV2(s_bindings.data(), s_stream, nullptr);

	// End graph capture
	cudaStreamEndCapture(s_stream, &s_graph);

	// Create executable graph
	cudaGraphInstantiate(&s_graphExec, s_graph, nullptr, nullptr, 0);

	s_graphInitialized = true;
	std::cout << "CUDA graph initialized successfully." << std::endl;
	return true;
}


std::vector<cv::Mat> TRTInference::performSegmentationInference(torch::Tensor img_tensor, int num_trials) {
	std::vector<cv::Mat> grayscale_images;

	// Check if engine is initialized
	if (!s_initialized || !s_engine || !s_context) {
		std::cerr << "ERROR: Engine not initialized. Call initializeTRTEngine first." << std::endl;
		return grayscale_images;
	}

	// Check that the input tensor is on CUDA
	TORCH_CHECK(img_tensor.device().is_cuda(), "Input tensor must be on CUDA");

	// Initialize CUDA graph if not already done
	if (!s_graphInitialized) {
		if (!initializeCudaGraph(img_tensor)) {
			std::cerr << "Failed to initialize CUDA graph." << std::endl;
			return grayscale_images;
		}
	}

	// Calculate the total number of elements in the input tensor
	int input_size = img_tensor.numel();

	// Copy input data to the persistent input device memory
	cudaMemcpyAsync(
		s_d_input,
		img_tensor.data_ptr<float>(),
		input_size * sizeof(float),
		cudaMemcpyDeviceToDevice,
		s_stream);

	// Launch the CUDA graph (much faster than regular enqueueV2)
	cudaGraphLaunch(s_graphExec, s_stream);

	// Synchronize to ensure computation is complete
	cudaStreamSynchronize(s_stream);

	// Copy the output tensor back to host
	float* last_h_output = s_h_outputs.back();
	void* last_d_output = s_d_outputs.back();
	int last_output_size = s_outputDims.d[0] * s_outputDims.d[1] * s_outputDims.d[2] * s_outputDims.d[3];
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float),
		cudaMemcpyDeviceToHost, s_stream);
	cudaStreamSynchronize(s_stream);

	// Process the output tensor with class remapping
	int batch = s_outputDims.d[0];     // should be 1
	int num_classes = s_outputDims.d[1];
	int height = s_outputDims.d[2];
	int width = s_outputDims.d[3];

	// Create a Torch tensor from the output data
	auto last_output_tensor = torch::from_blob(
		last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	// Get segmentation prediction via argmax along the class channel
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);

	// *** CLASS REMAPPING ***
	auto remapped_labels = torch::zeros_like(class_labels);

	// Map new model classes to original model classes
	remapped_labels = torch::where(class_labels == 7,
		torch::full_like(class_labels, 0),
		remapped_labels);

	remapped_labels = torch::where(class_labels == 11,
		torch::full_like(class_labels, 1),
		remapped_labels);

	remapped_labels = torch::where(class_labels == 21,
		torch::full_like(class_labels, 3),
		remapped_labels);

	remapped_labels = torch::where(class_labels == 23,
		torch::full_like(class_labels, 4),
		remapped_labels);

	remapped_labels = torch::where(class_labels == 26,
		torch::full_like(class_labels, 5),
		remapped_labels);

	// Apply scaling to the remapped labels
	int scale = 255 / 21;
	auto image_post = remapped_labels * scale;

	// Convert the single segmentation mask to a cv::Mat
	auto single_image_post = image_post[0].squeeze().to(torch::kU8);
	cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1,
		single_image_post.data_ptr<uint8_t>());
	grayscale_images.push_back(cv_img.clone());

	return grayscale_images;
}

void TRTInference::cleanupTRTEngine() {
	if (s_graphInitialized) {
		if (s_graphExec) {
			cudaGraphExecDestroy(s_graphExec);
			s_graphExec = nullptr;
		}
		if (s_graph) {
			cudaGraphDestroy(s_graph);
			s_graph = nullptr;
		}

		// Free persistent memory
		if (s_d_input) {
			cudaFree(s_d_input);
			s_d_input = nullptr;
		}

		for (float* h_output : s_h_outputs) {
			delete[] h_output;
		}
		s_h_outputs.clear();

		for (void* d_output : s_d_outputs) {
			cudaFree(d_output);
		}
		s_d_outputs.clear();

		if (s_stream) {
			cudaStreamDestroy(s_stream);
			s_stream = nullptr;
		}

		s_graphInitialized = false;
	}

	if (s_initialized) {
		if (s_context) {
			s_context->destroy();
			s_context = nullptr;
		}
		if (s_engine) {
			s_engine->destroy();
			s_engine = nullptr;
		}
		if (s_runtime) {
			s_runtime->destroy();
			s_runtime = nullptr;
		}
		s_initialized = false;
		std::cout << "TensorRT engine resources cleaned up." << std::endl;
	}
}

// Single Image Segmantion Inference
void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {

	std::cout << "STARTING measure_trt_performance" << std::endl;
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	// reads file in binary w/o preprocessing
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	float* h_input;
	int input_size = img_tensor.numel();  // Get total number of elements in the tensor
	cudaMallocHost((void**)&h_input, input_size * sizeof(float)); // Pinned memory

	int numBindings = engine->getNbBindings();
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;

	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		std::string outputTensorName = engine->getBindingName(i);
		outputTensorNames.push_back(outputTensorName);
	}

	// Extract the dimensions from img_tensor, assuming the input tensor format is 4D: NCHW
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);  // setting dimensions for binding 0, input tensor

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	// Create CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for input and add to bindings
	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	// Copy data from img_tensor to h_input
	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

	// Transfer input data to GPU with error check
	cudaError_t mallocErr = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (mallocErr != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(mallocErr) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// Handle dynamic dimensions
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

		cudaError_t status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
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

	// warm up
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	// nvtxRangePush("Inference");
	// Annotate using nvtx around inference
	for (int i = 0; i < num_trials; ++i) {
		// Run asynchronous inference using enqueueV2
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

	///////////////////////////////////////////
	///////////// POST PROCESSING /////////////
	///////////////////////////////////////////

	// Copy last output tensor back to host after all trials
	float* last_h_output = h_outputs.back();
	void* last_d_output = d_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	// cudaMemcpy(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Calculate stats for the last tensor
	float min_val = *min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// FIX THE MAX LINE
	int batch = outputDims.d[0]; // Assuming the channel dimension (classes) is at index 1
	int num_classes = outputDims.d[1]; // Assuming the channel dimension (classes) is at index 1
	int height = outputDims.d[2];
	int width = outputDims.d[3];

	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	// Debug: Print shape and content of the last output tensor
	std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		std::cout << last_output_tensor.size(i) << " ";
	}
	std::cout << std::endl;

	// max returns -> {max values}, {max indices}
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);


	int scale = 255 / 21;
	auto scale_tensor = torch::tensor(scale, class_labels.options());
	auto image_post = class_labels * scale;

	std::cout << "\nNumber of dimensions(image_post): " << image_post.dim() << std::endl;
	for (int i = 0; i < image_post.dim(); ++i) {
		std::cout << image_post.size(i) << " ";
	}
	std::cout << std::endl;

	// chw to hwc
	auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);

	std::cout << "Number of dimensions(permuted_img): " << permuted_img.dim() << std::endl;
	for (int i = 0; i < permuted_img.dim(); ++i) {
		std::cout << permuted_img.size(i) << " ";
	}

	cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());

	try {
		///////////////////////////////////////////////////////////////////
		cv::imwrite("pngOutput/trt_seg_output_scaled.png", cv_img);        // use your own path
		//////////////////////////////////////////////////////////////////

		cout << "Saved IMG: trt_seg_output_scaled" << endl;
	}
	catch (const cv::Exception& ex) {
		cerr << "Failed to save image trt_seg_output_scaled because ERROR:" << ex.what() << endl;
	}

	// Clean up
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
* Multiple Image Segmentation Inference
* This function is used in glow_effect::glow_effect_video function
*/
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul(const string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
    std::vector<cv::Mat> grayscale_images;  // for saving each gray-scaled imaghe

    std::cout << "STARTING measure_segmentation_trt_performance_mul" << std::endl;
    TRTGeneration::CustomLogger myLogger;
    IRuntime* runtime = createInferRuntime(myLogger);

    // reads file in binary w/o preprocessing
    ifstream planFile(trt_plan, ios::binary);
    vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

    ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    IExecutionContext* context = engine->createExecutionContext();
    if (!engine || !context) {
        cerr << "Failed to deserialize engine or create execution context." << endl;
        exit(EXIT_FAILURE);
    }

    float* h_input;
    int input_size = img_tensor_batch.numel();  // Get total number of elements in the tensor
    cudaMallocHost((void**)&h_input, input_size * sizeof(float)); // Pinned memory

    int numBindings = engine->getNbBindings();
    nvinfer1::Dims4 inputDims;
    nvinfer1::Dims outputDims;

    std::vector<int> outputBindingIndices;
    std::vector<std::string> outputTensorNames;

    for (int i = 1; i < numBindings; ++i) {
        outputBindingIndices.push_back(i);
        std::string outputTensorName = engine->getBindingName(i);
        outputTensorNames.push_back(outputTensorName);
    }

    // Extract the dimensions from img_tensor_batch, assuming the input tensor format is 4D: NCHW
    inputDims.d[0] = img_tensor_batch.size(0);
    inputDims.d[1] = img_tensor_batch.size(1);
    inputDims.d[2] = img_tensor_batch.size(2);
    inputDims.d[3] = img_tensor_batch.size(3);
    context->setBindingDimensions(0, inputDims);  // setting dimensions for binding 0, input tensor

    std::vector<void*> d_outputs;
    std::vector<float*> h_outputs;
    std::vector<void*> bindings;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory for input and add to bindings
    void* d_input;
    cudaMalloc(&d_input, input_size * sizeof(float));

    // Copy data from img_tensor_batch to h_input
    std::memcpy(h_input, img_tensor_batch.data_ptr<float>(), input_size * sizeof(float));

    // Transfer input data to GPU with error check
    cudaError_t mallocErr = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (mallocErr != cudaSuccess) {
        cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(mallocErr) << endl;
        exit(EXIT_FAILURE);
    }
    bindings.push_back(d_input);

    // Handle dynamic dimensions
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

        cudaError_t status = cudaMalloc(&d_output, outputSize * sizeof(float));
        if (status != cudaSuccess) {
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

    // warm up
    for (int i = 0; i < 10; ++i) {
        context->enqueueV2(bindings.data(), stream, nullptr);
    }

    cudaEventRecord(start, stream);
    // nvtxRangePush("Inference");
    // Annotate using nvtx around inference
    for (int i = 0; i < num_trials; ++i) {
        // Run asynchronous inference using enqueueV2
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

    ///////////////////////////////////////////
    ///////////// POST PROCESSING /////////////
    ///////////////////////////////////////////

    // Copy last output tensor back to host after all trials
    float* last_h_output = h_outputs.back();
    void* last_d_output = d_outputs.back();
    int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
    // cudaMemcpy(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate stats for the last tensor
    float min_val = *min_element(last_h_output, last_h_output + last_output_size);
    float max_val = *max_element(last_h_output, last_h_output + last_output_size);
    float avg_val = accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
    cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

    float average_latency = accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
    cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int batch = outputDims.d[0]; // Assuming the batch dimension is at index 0
    int num_classes = outputDims.d[1]; // Assuming the channel dimension (classes) is at index 1
    int height = outputDims.d[2];
    int width = outputDims.d[3];

    auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

    // Debug: Print shape and content of the last output tensor
    std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
    for (int i = 0; i < last_output_tensor.dim(); ++i) {
        std::cout << last_output_tensor.size(i) << " ";
    }
    std::cout << std::endl;

    // max returns -> {max values}, {max indices}
    auto max_out = torch::max(last_output_tensor, 1);
    auto class_labels = std::get<1>(max_out);

	// ================== DEBUGGING CODE START ==================
// Print the tensor shape and data type
	std::cout << "Class labels tensor shape: " << class_labels.sizes() << std::endl;
	std::cout << "Class labels data type: " << class_labels.dtype() << std::endl;

	// Count and print each unique class value
	// Move the tensor to CPU if needed for easy access
	auto cpu_labels = class_labels.to(torch::kCPU);
	auto flat_labels = cpu_labels.flatten();

	// Count class occurrences manually
	std::map<int64_t, int> class_counts;
	for (int i = 0; i < flat_labels.numel(); i++) {
		int64_t class_val = flat_labels[i].item<int64_t>();
		class_counts[class_val]++;
	}

	// Print class distribution
	std::cout << "Class distribution (pixels per class):" << std::endl;
	for (const auto& pair : class_counts) {
		int64_t class_idx = pair.first;
		int count = pair.second;
		float percentage = (float)count / flat_labels.numel() * 100;
		std::cout << "Class " << class_idx << ": " << count << " pixels ("
			<< percentage << "% of image)" << std::endl;

		// Create a binary mask for this class
		auto mask = (cpu_labels == class_idx).to(torch::kU8) * 255;
		if (batch == 1) {  // Handle single batch case
			auto mask_2d = mask[0];  // Get first batch item
			cv::Mat debug_mask(mask_2d.size(0), mask_2d.size(1), CV_8UC1,
				mask_2d.data_ptr<uint8_t>());

			// Save this class mask
			try {
				cv::imwrite("pngOutput/class_" + std::to_string(class_idx) + "_mask.png", debug_mask);
				std::cout << "Saved mask for class " << class_idx << std::endl;
			}
			catch (const cv::Exception& ex) {
				std::cerr << "Failed to save class mask: " << ex.what() << std::endl;
			}
		}
	}
	// ================== DEBUGGING CODE END ==================

    int scale = 255 / 21;
    auto scale_tensor = torch::tensor(scale, class_labels.options());
    auto image_post = class_labels * scale;

    // convert to HW format, and save each img to vector
    for (int i = 0; i < batch; ++i) {
        auto single_image_post = image_post[i].squeeze().to(torch::kU8);  // extract a single img, and convert to 8-bit
        cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());

        // add each gray-scale img to grayscale_images vector and save
        grayscale_images.push_back(cv_img.clone());  // use clone() to copy
        try {
            ///////////////////////////////////////////////////////////////////////////////////////////////
            cv::imwrite("pngOutput/trt_seg_output_scaled_" + std::to_string(i) + ".png", cv_img);         // move seg images to pngOutput directory
            //////////////////////////////////////////////////////////////////////////////////////////////
            
            cout << "Saved IMG: trt_seg_output_scaled_" + std::to_string(i) << endl;
        }
        catch (const cv::Exception& ex) {
            cerr << "Failed to save image trt_seg_output_scaled_" + std::to_string(i) + " because ERROR:" << ex.what() << endl;
        }
    }

    // cleanup
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

    return grayscale_images;  // return
}

std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_OPT(
	const std::string& trt_plan, torch::Tensor img_tensor, int num_trials)
{
	std::vector<cv::Mat> grayscale_images;  // for saving the grayscale segmentation image

	std::cout << "STARTING measure_segmentation_trt_performance_mul (Single Frame)" << std::endl;
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	// Read the TensorRT engine plan file (binary mode)
	std::ifstream planFile(trt_plan, std::ios::binary);
	std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
		std::istreambuf_iterator<char>());

	// Deserialize the engine and create an execution context
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		std::cerr << "Failed to deserialize engine or create execution context." << std::endl;
		exit(EXIT_FAILURE);
	}

	// ***********************************************************************
	// The input tensor is now expected to be a GPU tensor with batch size 1.
	// ***********************************************************************
	TORCH_CHECK(img_tensor.device().is_cuda(), "Input tensor must be on CUDA");

	// Calculate the total number of elements in the input tensor.
	int input_size = img_tensor.numel();

	// Create a CUDA stream.
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Allocate device memory for the input tensor.
	void* d_input;
	cudaError_t status = cudaMalloc(&d_input, input_size * sizeof(float));
	if (status != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for input." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Device-to-device copy directly from the GPU tensor's data pointer.
	cudaError_t copyErr = cudaMemcpyAsync(
		d_input,
		img_tensor.data_ptr<float>(),  // already on the GPU
		input_size * sizeof(float),
		cudaMemcpyDeviceToDevice,
		stream);
	if (copyErr != cudaSuccess) {
		std::cerr << "CUDA error (device-to-device cudaMemcpyAsync): "
			<< cudaGetErrorString(copyErr) << std::endl;
		exit(EXIT_FAILURE);
	}

	// Create the bindings vector and add the GPU input pointer.
	std::vector<void*> bindings;
	bindings.push_back(d_input);

	// Set up output bindings.
	int numBindings = engine->getNbBindings();
	std::vector<int> outputBindingIndices;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
	}

	// Extract dimensions from the input tensor (assuming shape [1, C, H, W]).
	nvinfer1::Dims4 inputDims;
	inputDims.d[0] = img_tensor.size(0);  // should be 1
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	// Allocate memory for outputs.
	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	nvinfer1::Dims outputDims;
	for (int i : outputBindingIndices) {
		outputDims = engine->getBindingDimensions(i);
		// Fix dynamic dimensions if necessary.
		for (int j = 0; j < outputDims.nbDims; ++j) {
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];  // allocate host memory for output
		void* d_output;
		status = cudaMalloc(&d_output, outputSize * sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "Device memory allocation for output failed" << std::endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	// Warm-up runs
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	// Timing inference using CUDA events.
	std::vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, stream);

	// Run inference trials.
	for (int i = 0; i < num_trials; ++i) {
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			std::cerr << "TensorRT enqueueV2 failed!" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	// Copy the last output tensor back to host after all trials.
	float* last_h_output = h_outputs.back();
	void* last_d_output = d_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Print statistics on the last output.
	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	std::cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val
		<< ", Avg: " << avg_val << std::endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	std::cout << "TRT - Average Latency over " << num_trials << " trials: "
		<< average_latency << " ms" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// ***********************************************************************
	// Now process the output tensor.
	// For a batch size of one, output dimensions are assumed to be [1, num_classes, H, W].
	// ***********************************************************************
	int batch = outputDims.d[0];       // should be 1
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];

	// Create a Torch tensor from the output data.
	auto last_output_tensor = torch::from_blob(
		last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	// Get segmentation prediction via argmax along the class channel.
	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);

	// *** CLASS REMAPPING ***
	// Create remapped labels tensor initialized with zeros
	auto remapped_labels = torch::zeros_like(class_labels);

	// Map new model classes to original model classes
	// Class 7 (27.37%) -> Class 0 (22.67%)
	remapped_labels = torch::where(class_labels == 7,
		torch::full_like(class_labels, 0),
		remapped_labels);

	// Class 11 (56.49%) -> Class 1 (37.54%)
	remapped_labels = torch::where(class_labels == 11,
		torch::full_like(class_labels, 1),
		remapped_labels);

	// Class 21 (11.09%) -> Class 3 (28.18%)
	remapped_labels = torch::where(class_labels == 21,
		torch::full_like(class_labels, 3),
		remapped_labels);

	// Class 23 (1.47%) -> Class 4 (2.96%)
	remapped_labels = torch::where(class_labels == 23,
		torch::full_like(class_labels, 4),
		remapped_labels);

	// Class 26 (3.59%) -> Class 5 (3.15%)
	remapped_labels = torch::where(class_labels == 26,
		torch::full_like(class_labels, 5),
		remapped_labels);
//	// ================== DEBUGGING CODE START ==================
//// Print the tensor shape and data type
//	std::cout << "Class labels tensor shape: " << class_labels.sizes() << std::endl;
//	std::cout << "Class labels data type: " << class_labels.dtype() << std::endl;
//
//	// Count and print each unique class value
//	// Move the tensor to CPU if needed for easy access
//	auto cpu_labels = class_labels.to(torch::kCPU);
//	auto flat_labels = cpu_labels.flatten();
//
//	// Count class occurrences manually
//	std::map<int64_t, int> class_counts;
//	for (int i = 0; i < flat_labels.numel(); i++) {
//		int64_t class_val = flat_labels[i].item<int64_t>();
//		class_counts[class_val]++;
//	}
//
//	// Print class distribution
//	std::cout << "Class distribution (pixels per class):" << std::endl;
//	for (const auto& pair : class_counts) {
//		int64_t class_idx = pair.first;
//		int count = pair.second;
//		float percentage = (float)count / flat_labels.numel() * 100;
//		std::cout << "Class " << class_idx << ": " << count << " pixels ("
//			<< percentage << "% of image)" << std::endl;
//
//		// Create a binary mask for this class
//		auto mask = (cpu_labels == class_idx).to(torch::kU8) * 255;
//		if (batch == 1) {  // Handle single batch case
//			auto mask_2d = mask[0];  // Get first batch item
//			cv::Mat debug_mask(mask_2d.size(0), mask_2d.size(1), CV_8UC1,
//				mask_2d.data_ptr<uint8_t>());
//
//			// Save this class mask
//			try {
//				cv::imwrite("pngOutput/class_" + std::to_string(class_idx) + "_mask.png", debug_mask);
//				std::cout << "Saved mask for class " << class_idx << std::endl;
//			}
//			catch (const cv::Exception& ex) {
//				std::cerr << "Failed to save class mask: " << ex.what() << std::endl;
//			}
//		}
//	}
//	// ================== DEBUGGING CODE END ==================
	int scale = 255 / 21;
	auto image_post = remapped_labels * scale;

	// Convert the single segmentation mask to a cv::Mat.
	auto single_image_post = image_post[0].squeeze().to(torch::kU8);
	cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1,
		single_image_post.data_ptr<uchar>());
	grayscale_images.push_back(cv_img.clone());
	try {
		cv::imwrite("pngOutput/trt_seg_output_scaled.png", cv_img);
		std::cout << "Saved IMG: trt_seg_output_scaled.png" << std::endl;
	}
	catch (const cv::Exception& ex) {
		std::cerr << "Failed to save image trt_seg_output_scaled due to ERROR:" << ex.what() << std::endl;
	}

	// Cleanup: free allocated host and device memory, destroy context, engine, runtime, and stream.
	cudaFree(d_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
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

	// -----------------------------
	// Create TensorRT runtime and deserialize the engine from the plan file.
	// -----------------------------
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!engine) {
		std::cerr << "Failed to deserialize engine in concurrent segmentation." << std::endl;
		exit(EXIT_FAILURE);
	}

	// -----------------------------
	// Determine batch and thread parameters.
	// -----------------------------
	int totalBatch = img_tensor_batch.size(0);  // Total number of images.
	int numThreads = 2;                           // Fixed number of threads.
	int subBatch = (totalBatch + numThreads - 1) / numThreads;  // Images per thread.

	// allResults will store the segmentation output for each image.
	std::vector<cv::Mat> allResults(totalBatch);
	std::mutex resultMutex;  // Mutex to protect shared results vector.
	std::vector<std::thread> threads;

	// -----------------------------
	// Launch threads to process sub-batches concurrently.
	// -----------------------------
	for (int t = 0; t < numThreads; ++t) {
		threads.emplace_back([&, t]() {
			// Create execution context for this thread.
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Failed to create execution context for thread " << t << std::endl;
				return;
			}

			// Create a CUDA stream with non-blocking flags.
			cudaStream_t stream;
			checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

			// Calculate sub-batch indices for this thread.
			int startIdx = t * subBatch;
			int endIdx = std::min(startIdx + subBatch, totalBatch);
			int validCount = endIdx - startIdx;  // Number of valid images in this sub-batch.
			if (validCount <= 0)
				return;

			// -----------------------------
			// Extract sub-batch from the global image tensor.
			// -----------------------------
			torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);
			// Remove extra singleton dimension if present (e.g., [N,1,3,384,384] -> [N,3,384,384]).
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

			// -----------------------------
			// Allocate host and device memory for input data.
			// -----------------------------
			int inputSize = subTensor.numel();
			float* h_input = nullptr;
			checkCudaErrors(cudaMallocHost((void**)&h_input, inputSize * sizeof(float)));
			std::memcpy(h_input, subTensor.data_ptr<float>(), inputSize * sizeof(float));

			void* d_input = nullptr;
			checkCudaErrors(cudaMalloc(&d_input, inputSize * sizeof(float)));
			checkCudaErrors(cudaMemcpyAsync(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));

			// Set input dimensions for the context.
			nvinfer1::Dims4 inputDims;
			inputDims.d[0] = subTensor.size(0);
			inputDims.d[1] = subTensor.size(1);
			inputDims.d[2] = subTensor.size(2);
			inputDims.d[3] = subTensor.size(3);
			context->setBindingDimensions(0, inputDims);

			// -----------------------------
			// Prepare output buffers.
			// -----------------------------
			std::vector<void*> bindings;
			bindings.push_back(d_input);  // Binding index 0: Input buffer.
			int numBindings = engine->getNbBindings();
			std::vector<float*> h_outputs;  // Host memory for outputs.
			std::vector<void*> d_outputs;   // Device memory for outputs.
			nvinfer1::Dims4 outputDims;
			for (int i = 1; i < numBindings; ++i) {
				// Retrieve binding dimensions for each output.
				nvinfer1::Dims dims = context->getBindingDimensions(i);
				outputDims.nbDims = dims.nbDims;
				for (int j = 0; j < dims.nbDims; ++j) {
					outputDims.d[j] = dims.d[j];
					if (outputDims.d[j] < 0)
						outputDims.d[j] = inputDims.d[j];  // Fallback to input dimensions if negative.
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

			// -----------------------------
			// Perform optional warm-up runs.
			// -----------------------------
			for (int i = 0; i < 3; ++i) {
				context->enqueueV2(bindings.data(), stream, nullptr);
			}

			// -----------------------------
			// Enqueue inference and check for errors.
			// -----------------------------
			if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
				std::cerr << "TensorRT enqueueV2 failed in thread " << t << std::endl;
				exit(EXIT_FAILURE);
			}

			// -----------------------------
			// Copy inference results from device to host.
			// -----------------------------
			int outSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
			float* lastOutput = h_outputs.back();
			checkCudaErrors(cudaMemcpyAsync(lastOutput, d_outputs.back(), outSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
			cudaStreamSynchronize(stream);  // Ensure all operations complete.

			// -----------------------------
			// Post-process the output tensor.
			// -----------------------------
			auto outputTensor = torch::from_blob(lastOutput, { outputDims.d[0], outputDims.d[1], outputDims.d[2], outputDims.d[3] }, torch::kFloat32);
			auto maxOut = torch::max(outputTensor, 1);  // Find maximum values along the channel dimension.
			auto segMask = std::get<1>(maxOut);  // Expected shape: [4, H, W], where each element is the class index.
			int scaleFactor = 255 / 21;  // Scale factor to map class indices to an 8-bit range.
			auto imagePost = segMask * scaleFactor;

			// -----------------------------
			// Convert tensor outputs to cv::Mat and handle padded outputs.
			// -----------------------------
			std::vector<cv::Mat> localResults;
			for (int i = 0; i < validCount; ++i) {
				auto single = imagePost[i].to(torch::kU8);  // Convert to unsigned 8-bit.
				cv::Mat result(single.size(0), single.size(1), CV_8UC1, single.data_ptr<uchar>());
				localResults.push_back(result.clone());
			}

			// -----------------------------
			// Safely update the global results vector.
			// -----------------------------
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				for (int i = 0; i < validCount; ++i) {
					allResults[startIdx + i] = localResults[i];
				}
			}

			// -----------------------------
			// Free allocated host and device memory, destroy CUDA stream and context.
			// -----------------------------
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

	// Wait for all threads to complete execution.
	for (auto& t : threads)
		t.join();

	// Destroy the engine and runtime to free resources.
	engine->destroy();
	runtime->destroy();

	return allResults;
}

//--------------------------------------------------------------------------
// Concurrent Segmentation with CUDA Graph
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent_graph(const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {

	std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent_graph (Hybrid CUDA Graph approach)" << std::endl;

	// Create TensorRT runtime and deserialize the engine
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	std::cout << "Loaded engine size: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;

	auto start_time = std::chrono::high_resolution_clock::now();
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Deserialization required " << duration.count() << " microseconds." << std::endl;

	if (!engine) {
		std::cerr << "Failed to deserialize engine in graph segmentation." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Setup for multi-threaded processing
	int totalBatch = img_tensor_batch.size(0);
	int numThreads = 2;
	int subBatch = (totalBatch + numThreads - 1) / numThreads;
	std::vector<cv::Mat> allResults(totalBatch);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	// Launch threads to process sub-batches concurrently
	for (int t = 0; t < numThreads; ++t) {
		threads.emplace_back([&, t]() {
			// Create execution context for this thread
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Failed to create execution context for thread " << t << std::endl;
				return;
			}

			// Create three streams: one for pre-processing, one for inference, one for post-processing
			cudaStream_t preStream, inferStream, postStream;
			checkCudaErrors(cudaStreamCreateWithFlags(&preStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));

			// Calculate sub-batch indices
			int startIdx = t * subBatch;
			int endIdx = std::min(startIdx + subBatch, totalBatch);
			int validCount = endIdx - startIdx;
			if (validCount <= 0)
				return;

			// Extract sub-batch from the global image tensor
			torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);

			// Remove extra singleton dimensions if needed
			if ((subTensor.dim() == 5 && subTensor.size(1) == 1) ||
				(subTensor.dim() == 4 && subTensor.size(1) == 1))
			{
				subTensor = subTensor.squeeze(1);
			}

			// Pad the sub-batch to have exactly 4 images if needed
			if (subTensor.size(0) < 4) {
				int pad = 4 - subTensor.size(0);
				torch::Tensor lastFrame = subTensor[subTensor.size(0) - 1].unsqueeze(0);
				torch::Tensor padTensor = lastFrame.repeat({ pad, 1, 1, 1 });
				subTensor = torch::cat({ subTensor, padTensor }, 0);
			}

			// Set input dimensions
			nvinfer1::Dims4 inputDims;
			inputDims.d[0] = subTensor.size(0);
			inputDims.d[1] = subTensor.size(1);
			inputDims.d[2] = subTensor.size(2);
			inputDims.d[3] = subTensor.size(3);
			context->setBindingDimensions(0, inputDims);

			// Allocate memory for input and output
			int inputSize = subTensor.numel();
			float* h_input = nullptr;
			void* d_input = nullptr;
			checkCudaErrors(cudaMallocHost((void**)&h_input, inputSize * sizeof(float)));
			checkCudaErrors(cudaMalloc(&d_input, inputSize * sizeof(float)));

			// Copy input data to host pinned memory
			std::memcpy(h_input, subTensor.data_ptr<float>(), inputSize * sizeof(float));

			// Setup bindings and allocate output memory
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
				void* d_output = nullptr;
				checkCudaErrors(cudaMallocHost((void**)&h_output, outputSize * sizeof(float)));
				checkCudaErrors(cudaMalloc(&d_output, outputSize * sizeof(float)));
				h_outputs.push_back(h_output);
				d_outputs.push_back(d_output);
				bindings.push_back(d_output);
			}

			// Allocate device memory for post-processing
			int batch = outputDims.d[0];
			int height = outputDims.d[2];
			int width = outputDims.d[3];
			unsigned char* d_argmax_output = nullptr;
			checkCudaErrors(cudaMalloc(&d_argmax_output, batch * height * width * sizeof(unsigned char)));

			// Pre-processing: Copy input from host to device (not part of the graph)
			checkCudaErrors(cudaMemcpyAsync(d_input, h_input, inputSize * sizeof(float),
				cudaMemcpyHostToDevice, preStream));
			checkCudaErrors(cudaStreamSynchronize(preStream));

			// Setup timing
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			// Declare graph objects outside the try block
			cudaGraph_t postprocessGraph = nullptr;
			cudaGraphExec_t postprocessGraphExec = nullptr;
			bool useGraph = false;

			// To fix the compilation error with goto, we'll declare the results vector here
			std::vector<cv::Mat> localResults;
			unsigned char* h_argmax_output = nullptr;

			// First perform a regular inference run for warmup
			// This is always done regardless of graph capture success
			for (int i = 0; i < 2; ++i) {
				// Run TensorRT inference
				if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
					std::cerr << "TensorRT enqueueV2 failed during warmup" << std::endl;
					goto cleanup; // Jump to resource cleanup
				}
				checkCudaErrors(cudaStreamSynchronize(inferStream));
			}

			// Now try to create a graph for post-processing operations
			try {
				// Create a graph for post-processing kernels
				checkCudaErrors(cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed));

				// Call our custom kernel for argmax operation
				launchArgmaxKernel(
					static_cast<float*>(d_outputs.back()),
					d_argmax_output,
					batch,
					outputDims.d[1], // num_classes
					height,
					width,
					postStream
				);

				checkCudaErrors(cudaStreamEndCapture(postStream, &postprocessGraph));
				checkCudaErrors(cudaGraphInstantiate(&postprocessGraphExec, postprocessGraph, nullptr, nullptr, 0));

				// If we get here, graph capture for post-processing was successful
				useGraph = true;
				std::cout << "Thread " << t << " successfully created post-processing graph" << std::endl;
			}
			catch (const std::exception& e) {
				std::cerr << "CUDA Graph capture failed: " << e.what() << std::endl;
				std::cerr << "Falling back to regular execution..." << std::endl;
				useGraph = false;

				// Clean up any partial graph resources
				if (postprocessGraph) {
					cudaGraphDestroy(postprocessGraph);
					postprocessGraph = nullptr;
				}
				if (postprocessGraphExec) {
					cudaGraphExecDestroy(postprocessGraphExec);
					postprocessGraphExec = nullptr;
				}
			}

			// Execute inference timing
			cudaEventRecord(start, inferStream);

			// For TensorRT inference, we always use regular execution since it's not compatible with graph capture
			if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
				std::cerr << "TensorRT enqueueV2 failed" << std::endl;
				goto cleanup; // Jump to resource cleanup
			}
			checkCudaErrors(cudaStreamSynchronize(inferStream));

			// Execute post-processing (either with graph or regular method)
			if (useGraph && postprocessGraphExec) {
				checkCudaErrors(cudaGraphLaunch(postprocessGraphExec, postStream));
				checkCudaErrors(cudaStreamSynchronize(postStream));
			}
			else {
				// Fall back to regular kernel launch if graph capture failed
				launchArgmaxKernel(
					static_cast<float*>(d_outputs.back()),
					d_argmax_output,
					batch,
					outputDims.d[1], // num_classes
					height,
					width,
					postStream
				);
				checkCudaErrors(cudaStreamSynchronize(postStream));
			}

			// Copy results from device to host
			h_argmax_output = new unsigned char[batch * height * width];
			checkCudaErrors(cudaMemcpyAsync(
				h_argmax_output,
				d_argmax_output,
				batch * height * width * sizeof(unsigned char),
				cudaMemcpyDeviceToHost,
				postStream
			));
			checkCudaErrors(cudaStreamSynchronize(postStream));

			cudaEventRecord(stop, inferStream);
			checkCudaErrors(cudaStreamSynchronize(inferStream));

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "Thread " << t << " execution time: " << milliseconds << " ms"
				<< (useGraph ? " (with partial CUDA Graph)" : " (without CUDA Graph)") << std::endl;

			// Convert to OpenCV Mats and update results
			for (int i = 0; i < validCount; ++i) {
				cv::Mat result(height, width, CV_8UC1);
				// Copy the segmentation result for this batch item
				std::memcpy(
					result.data,
					h_argmax_output + (i * height * width),
					height * width * sizeof(unsigned char)
				);
				localResults.push_back(result.clone());
			}

			// Update shared results vector with thread-local results
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				for (int i = 0; i < validCount; ++i) {
					allResults[startIdx + i] = localResults[i];
				}
			}

		cleanup:
			// Clean up temporary host memory
			if (h_argmax_output) {
				delete[] h_argmax_output;
			}

			// Clean up resources
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			if (postprocessGraphExec) {
				cudaGraphExecDestroy(postprocessGraphExec);
			}
			if (postprocessGraph) {
				cudaGraphDestroy(postprocessGraph);
			}

			cudaFree(d_argmax_output);
			cudaFreeHost(h_input);
			cudaFree(d_input);
			for (auto ptr : h_outputs) {
				cudaFreeHost(ptr);
			}
			for (auto dptr : d_outputs) {
				cudaFree(dptr);
			}
			cudaStreamDestroy(preStream);
			cudaStreamDestroy(inferStream);
			cudaStreamDestroy(postStream);
			context->destroy();
			});
	}

	// Wait for all threads to complete
	for (auto& t : threads) {
		t.join();
	}

	// Clean up global resources
	engine->destroy();
	runtime->destroy();

	return allResults;
}

//--------------------------------------------------------------------------------------
// Processes multiple images in parallel using a single-batch TRT model with CUDA Graph
//--------------------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel(const std::string& trt_plan, const std::vector<torch::Tensor>& img_tensors, int num_streams) {

	std::cout << "Starting optimized parallel single-batch segmentation with post-processing CUDA Graph acceleration" << std::endl;

	// Number of images to process
	int num_images = img_tensors.size();
	if (num_images == 0) {
		return {};
	}

	// Create the TensorRT runtime and load the engine
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	std::ifstream planFile(trt_plan, std::ios::binary);
	if (!planFile.is_open()) {
		std::cerr << "Error: Could not open plan file: " << trt_plan << std::endl;
		return {};
	}

	std::vector<char> plan((std::istreambuf_iterator<char>(planFile)), std::istreambuf_iterator<char>());
	std::cout << "Loaded single-batch plan file: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!engine) {
		std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
		runtime->destroy();
		return {};
	}

	// Results container
	std::vector<cv::Mat> results(num_images);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	// Calculate images per worker thread
	int images_per_thread = (num_images + num_streams - 1) / num_streams;

	// Performance metrics for reporting
	std::vector<double> processing_times(num_streams, 0.0);
	std::vector<int> frames_processed(num_streams, 0);
	std::vector<bool> graph_usage(num_streams, false);

	// Launch parallel worker threads
	for (int t = 0; t < num_streams; ++t) {
		threads.emplace_back([&, t]() {
			// Calculate the range of images for this worker
			int start_idx = t * images_per_thread;
			int end_idx = std::min(start_idx + images_per_thread, num_images);

			if (start_idx >= num_images) {
				return; // No images for this worker
			}

			// Create execution context for this worker
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Error: Failed to create execution context for worker " << t << std::endl;
				return;
			}

			// Create CUDA streams for this worker - separate streams for inference and post-processing
			cudaStream_t inferStream, postStream;
			checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));

			// Timing variables
			auto worker_start_time = std::chrono::high_resolution_clock::now();
			int local_frames_processed = 0;

			// Variables for post-processing CUDA graph
			cudaGraph_t postprocessGraph = nullptr;
			cudaGraphExec_t postprocessGraphExec = nullptr;
			bool postGraphCaptured = false;

			// Process each image assigned to this worker
			for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
				const torch::Tensor& img_tensor = img_tensors[img_idx];

				// Verify tensor dimensions and that it resides on GPU
				if (img_tensor.dim() != 4 || img_tensor.size(0) != 1) {
					std::cerr << "Error: Invalid tensor dimensions for image " << img_idx
						<< ". Expected 4D tensor with batch size 1." << std::endl;
					continue;
				}
				if (!img_tensor.is_cuda()) {
					std::cerr << "Error: img_tensor for image " << img_idx << " is not on GPU." << std::endl;
					continue;
				}

				try {
					// === SET UP INFERENCE INPUT ===

					// Set input dimensions (always batch size 1 for single-batch model)
					nvinfer1::Dims4 inputDims;
					inputDims.d[0] = 1;
					inputDims.d[1] = img_tensor.size(1);
					inputDims.d[2] = img_tensor.size(2);
					inputDims.d[3] = img_tensor.size(3);
					context->setBindingDimensions(0, inputDims);

					if (!context->allInputDimensionsSpecified()) {
						std::cerr << "Error: Not all input dimensions were specified for image " << img_idx << std::endl;
						continue;
					}

					// Since the tensor is already on the GPU, we simply use its device pointer directly.
					size_t input_size = img_tensor.numel();
					float* d_input = const_cast<float*>(img_tensor.data_ptr<float>());

					// Set up bindings and allocate output memory
					std::vector<void*> bindings = { d_input };
					std::vector<void*> d_outputs;
					std::vector<float*> h_outputs;
					nvinfer1::Dims outputDims;

					// Setup output bindings - always get output binding info after setting input dimensions
					for (int i = 1; i < engine->getNbBindings(); ++i) {
						outputDims = context->getBindingDimensions(i);
						int outputSize = 1;
						for (int j = 0; j < outputDims.nbDims; ++j) {
							outputSize *= outputDims.d[j];
						}

						float* h_output = nullptr;
						void* d_output = nullptr;

						cudaError_t cuda_error = cudaMallocHost((void**)&h_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating host output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							// No need to free d_input since it's managed by the tensor.
							break;
						}

						cuda_error = cudaMalloc(&d_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating device output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							cudaFreeHost(h_output);
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							break;
						}

						h_outputs.push_back(h_output);
						d_outputs.push_back(d_output);
						bindings.push_back(d_output);
					}

					// Check if we hit an error in the binding setup loop
					if (h_outputs.size() != engine->getNbBindings() - 1) {
						continue; // Skip to next image if memory allocation failed
					}

					// Get output dimensions for post-processing
					int batch = outputDims.d[0]; // Should be 1
					int num_classes = outputDims.d[1];
					int height = outputDims.d[2];
					int width = outputDims.d[3];

					// Allocate memory for segmentation mask output
					unsigned char* d_argmax_output = nullptr;
					cudaError_t cuda_error = cudaMalloc(&d_argmax_output, height * width * sizeof(unsigned char));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating argmax output memory: " << cudaGetErrorString(cuda_error) << std::endl;
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						continue;
					}

					// === RUN TENSORRT INFERENCE (NO GRAPH CAPTURE) ===
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, inferStream);

					// Run TensorRT inference - NOT in a CUDA graph since it's not supported
					if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
						std::cerr << "Error: TensorRT enqueueV2 failed for image " << img_idx << std::endl;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						continue;
					}

					// Wait for inference to complete
					cudaStreamSynchronize(inferStream);

					// === POST-PROCESSING WITH CUDA GRAPH ===
					// Try to create and execute a CUDA graph for post-processing only

					// Only capture the graph on the first successful run
					if (!postGraphCaptured) {
						try {
							// Start graph capture for post-processing only
							cuda_error = cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to begin graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							// Add argmax kernel to the graph
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);

							// End capture and instantiate the graph
							cuda_error = cudaStreamEndCapture(postStream, &postprocessGraph);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to end graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							cuda_error = cudaGraphInstantiate(&postprocessGraphExec, postprocessGraph, nullptr, nullptr, 0);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to instantiate graph: ") +
									cudaGetErrorString(cuda_error));
							}

							postGraphCaptured = true;
							graph_usage[t] = true;
							std::cout << "Worker " << t << ": Successfully created post-processing graph" << std::endl;
						}
						catch (const std::exception& e) {
							std::cerr << "Worker " << t << ": CUDA Graph capture for post-processing failed: "
								<< e.what() << std::endl;
							std::cerr << "Falling back to normal execution mode for post-processing" << std::endl;

							// Clean up any partial graph resources
							if (postprocessGraph) {
								cudaGraphDestroy(postprocessGraph);
								postprocessGraph = nullptr;
							}
							if (postprocessGraphExec) {
								cudaGraphExecDestroy(postprocessGraphExec);
								postprocessGraphExec = nullptr;
							}
						}
					}

					// Execute post-processing (with graph if available, or regular execution)
					if (postGraphCaptured && postprocessGraphExec) {
						// Execute the captured post-processing graph
						cuda_error = cudaGraphLaunch(postprocessGraphExec, postStream);
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error launching post-processing graph: "
								<< cudaGetErrorString(cuda_error) << std::endl;
							// Fall back to regular kernel launch
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);
						}
					}
					else {
						// Fall back to regular kernel launch if graph not available
						launchArgmaxKernel(
							static_cast<float*>(d_outputs.back()),
							d_argmax_output,
							1, // batch size is 1
							num_classes,
							height,
							width,
							postStream
						);
					}

					// Wait for post-processing to complete
					cudaStreamSynchronize(postStream);

					// Copy results back to host
					unsigned char* h_argmax_output = new unsigned char[height * width];
					cuda_error = cudaMemcpyAsync(h_argmax_output, d_argmax_output,
						height * width * sizeof(unsigned char),
						cudaMemcpyDeviceToHost, postStream);
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error copying results to host: " << cudaGetErrorString(cuda_error) << std::endl;
						delete[] h_argmax_output;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						continue;
					}
					cudaStreamSynchronize(postStream);

					// Record end timing
					cudaEventRecord(stop, inferStream);
					cudaStreamSynchronize(inferStream);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);

					// Create OpenCV Mat from the segmentation mask
					cv::Mat result(height, width, CV_8UC1);
					std::memcpy(result.data, h_argmax_output, height * width * sizeof(unsigned char));

					// Update the results array
					{
						std::lock_guard<std::mutex> lock(resultMutex);
						results[img_idx] = result.clone();
					}

					// Update local counters
					local_frames_processed++;

					// Cleanup per-image resources (note: we do not free d_input since it comes from the tensor)
					delete[] h_argmax_output;
					cudaFree(d_argmax_output);
					cudaEventDestroy(start);
					cudaEventDestroy(stop);
					for (auto ptr : h_outputs) {
						cudaFreeHost(ptr);
					}
					for (auto dptr : d_outputs) {
						cudaFree(dptr);
					}
				}
				catch (const std::exception& e) {
					std::cerr << "Error processing image " << img_idx << ": " << e.what() << std::endl;
					// Continue with next image
				}
			}

			// Calculate total processing time for this worker
			auto worker_end_time = std::chrono::high_resolution_clock::now();
			double total_seconds = std::chrono::duration<double>(worker_end_time - worker_start_time).count();

			// Update worker statistics
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				processing_times[t] = total_seconds;
				frames_processed[t] = local_frames_processed;
			}

			// Clean up worker resources
			if (postprocessGraphExec) {
				cudaGraphExecDestroy(postprocessGraphExec);
			}
			if (postprocessGraph) {
				cudaGraphDestroy(postprocessGraph);
			}
			cudaStreamDestroy(inferStream);
			cudaStreamDestroy(postStream);
			context->destroy();
			});
	}

	// Wait for all worker threads to complete
	for (auto& t : threads) {
		t.join();
	}

	// Summarize performance statistics
	std::cout << "\n=== Performance Summary ===" << std::endl;
	std::cout << "Total images processed: " << num_images << std::endl;

	double total_processing_time = 0.0;
	int total_processed = 0;
	int graph_workers = 0;

	for (int t = 0; t < num_streams; ++t) {
		std::cout << "Worker " << t << ": " << frames_processed[t] << " frames in "
			<< processing_times[t] << " seconds";
		if (frames_processed[t] > 0) {
			double fps = frames_processed[t] / processing_times[t];
			std::cout << " (" << fps << " fps)";
		}
		std::cout << (graph_usage[t] ? " [with CUDA Graph]" : " [without CUDA Graph]") << std::endl;
		total_processing_time += processing_times[t];
		total_processed += frames_processed[t];
		if (graph_usage[t]) graph_workers++;
	}
	std::cout << "Average processing time per worker: " << total_processing_time / num_streams << " seconds" << std::endl;
	std::cout << "Effective overall throughput: " << num_images / (total_processing_time / num_streams) << " fps" << std::endl;
	std::cout << "Workers using CUDA Graph: " << graph_workers << " of " << num_streams << std::endl;
	std::cout << "============================" << std::endl;

	// Clean up global resources
	engine->destroy();
	runtime->destroy();

	return results;
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
