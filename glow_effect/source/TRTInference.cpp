/**
 * @file TRTInference.cpp
 * @brief Implementation of TensorRT inference routines for segmentation and super-resolution.
 *
 * This file provides functions to measure inference performance, execute batched or single-image
 * segmentation, and process super-resolution outputs. The implementation includes various
 * optimization techniques such as CUDA graphs, parallel processing with multiple streams,
 * and efficient memory management.
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
 
 // External parameters defined in control_gui.cpp
 extern int param_KeyLevel;
 extern int param_KeyScale;
 extern int default_scale;
 
 // Initialize static class members
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
 
 /**
  * @brief Structure holding engine, context, and persistent resources for inference
  * 
  * This structure maintains the state for a TensorRT engine instance including its execution context
  * and associated CUDA resources like streams, events, and memory buffers.
  */
 struct EngineContextPair {
	 /// TensorRT engine for model execution
	 nvinfer1::ICudaEngine* engine;
	 /// Execution context for the engine
	 nvinfer1::IExecutionContext* context;
	 /// Persistent device memory for output tensors
	 std::vector<void*> persistent_d_outputs;
	 /// Persistent host memory for output tensors
	 std::vector<float*> persistent_h_outputs;
	 /// Dimensions of the output tensors
	 std::vector<nvinfer1::Dims> persistent_output_dims;
	 /// Flag indicating if persistent buffers are allocated
	 bool buffersAllocated = false;
	 /// CUDA stream for inference operations
	 cudaStream_t inferStream = nullptr;
	 /// CUDA stream for post-processing operations
	 cudaStream_t postStream = nullptr;
	 /// CUDA event for inference timing (start)
	 cudaEvent_t startEvent = nullptr;
	 /// CUDA event for inference timing (stop)
	 cudaEvent_t stopEvent = nullptr;
	 /// Flag indicating if streams and events are initialized
	 bool streamsEventsInitialized = false;
 };
 
 // Global resources for engine pooling
 static std::mutex enginePoolMutex;
 static std::vector<EngineContextPair> persistentEnginePool;
 
 //-----------------------------------------------------------------------------
 // Core Engine Functions
 //-----------------------------------------------------------------------------
 
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
 
	 // Create a persistent execution context
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
	 // Run one inference call to finish any lazy initialization
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
 
 //-----------------------------------------------------------------------------
 // Single-Image Segmentation Inference
 //-----------------------------------------------------------------------------
 
 void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {
	 std::cout << "STARTING measure_trt_performance" << std::endl;
 
	 // Initialize TensorRT runtime and engine
	 TRTGeneration::CustomLogger myLogger;
	 IRuntime* runtime = createInferRuntime(myLogger);
 
	 // Read the TensorRT engine plan file (binary mode)
	 ifstream planFile(trt_plan, ios::binary);
	 vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
 
	 // Deserialize the engine and create execution context
	 ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	 IExecutionContext* context = engine->createExecutionContext();
	 if (!engine || !context) {
		 cerr << "Failed to deserialize engine or create execution context." << endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Allocate pinned memory for input data
	 float* h_input;
	 int input_size = img_tensor.numel();
	 cudaMallocHost((void**)&h_input, input_size * sizeof(float));
 
	 // Set binding dimensions and identify output bindings
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
 
	 // Extract dimensions from input tensor (NCHW format)
	 inputDims.d[0] = img_tensor.size(0);
	 inputDims.d[1] = img_tensor.size(1);
	 inputDims.d[2] = img_tensor.size(2);
	 inputDims.d[3] = img_tensor.size(3);
	 context->setBindingDimensions(0, inputDims);
 
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
 
	 // Handle dynamic dimensions and allocate output memory
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
 
	 // Setup timing
	 vector<float> latencies;
	 cudaEvent_t start, stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
 
	 // Warm-up runs to eliminate initialization overhead
	 for (int i = 0; i < 10; ++i) {
		 context->enqueueV2(bindings.data(), stream, nullptr);
	 }
 
	 // Perform timed inference runs
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
 
	 //--------------------------------------------------
	 // Post-processing
	 //--------------------------------------------------
 
	 // Copy last output tensor back to host after all trials
	 float* last_h_output = h_outputs.back();
	 void* last_d_output = d_outputs.back();
	 int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	 cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	 cudaStreamSynchronize(stream);
 
	 // Calculate statistics for the output tensor
	 float min_val = *min_element(last_h_output, last_h_output + last_output_size);
	 float max_val = *max_element(last_h_output, last_h_output + last_output_size);
	 float avg_val = accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	 cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;
 
	 float average_latency = accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	 cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	 cudaEventDestroy(start);
	 cudaEventDestroy(stop);
 
	 // Extract output dimensions
	 int batch = outputDims.d[0];
	 int num_classes = outputDims.d[1];
	 int height = outputDims.d[2];
	 int width = outputDims.d[3];
 
	 auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);
 
	 // Debug: Print shape of the output tensor
	 std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
	 for (int i = 0; i < last_output_tensor.dim(); ++i) {
		 std::cout << last_output_tensor.size(i) << " ";
	 }
	 std::cout << std::endl;
 
	 // Get class predictions using argmax along class dimension
	 auto max_out = torch::max(last_output_tensor, 1);
	 auto class_labels = std::get<1>(max_out);
 
	 // Scale class indices to 8-bit range for visualization
	 int scale = 255 / 21;
	 auto scale_tensor = torch::tensor(scale, class_labels.options());
	 auto image_post = class_labels * scale;
 
	 std::cout << "\nNumber of dimensions(image_post): " << image_post.dim() << std::endl;
	 for (int i = 0; i < image_post.dim(); ++i) {
		 std::cout << image_post.size(i) << " ";
	 }
	 std::cout << std::endl;
 
	 // Permute tensor from CHW to HWC format for OpenCV
	 auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);
 
	 std::cout << "Number of dimensions(permuted_img): " << permuted_img.dim() << std::endl;
	 for (int i = 0; i < permuted_img.dim(); ++i) {
		 std::cout << permuted_img.size(i) << " ";
	 }
 
	 // Create OpenCV Mat from tensor data
	 cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());
 
	 try {
		 // Save the segmentation output as an image
		 cv::imwrite("pngOutput/trt_seg_output_scaled.png", cv_img);
		 cout << "Saved IMG: trt_seg_output_scaled" << endl;
	 }
	 catch (const cv::Exception& ex) {
		 cerr << "Failed to save image trt_seg_output_scaled because ERROR:" << ex.what() << endl;
	 }
 
	 // Clean up resources
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
 
 //-----------------------------------------------------------------------------
 // Multi-Image Segmentation Inference
 //-----------------------------------------------------------------------------
 
 /**
  * @brief Performs segmentation inference on a batch of images.
  * 
  * This function is used in glow_effect::glow_effect_video function to process
  * multiple frames in a batch.
  * 
  * @param trt_plan Path to the TensorRT plan file
  * @param img_tensor_batch Batch of input images as a single tensor
  * @param num_trials Number of inference trials to run
  * @return Vector of grayscale segmentation masks
  */
 std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul(
	 const string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
	 
	 std::vector<cv::Mat> grayscale_images;  // For storing each grayscale segmentation mask
 
	 std::cout << "STARTING measure_segmentation_trt_performance_mul" << std::endl;
	 
	 // Initialize TensorRT runtime and engine
	 TRTGeneration::CustomLogger myLogger;
	 IRuntime* runtime = createInferRuntime(myLogger);
 
	 // Read the TensorRT engine plan file (binary mode)
	 ifstream planFile(trt_plan, ios::binary);
	 vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
 
	 // Deserialize the engine and create execution context
	 ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	 IExecutionContext* context = engine->createExecutionContext();
	 if (!engine || !context) {
		 cerr << "Failed to deserialize engine or create execution context." << endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Allocate pinned memory for input data
	 float* h_input;
	 int input_size = img_tensor_batch.numel();
	 cudaMallocHost((void**)&h_input, input_size * sizeof(float));
 
	 // Set binding dimensions and identify output bindings
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
 
	 // Extract dimensions from input tensor batch (NCHW format)
	 inputDims.d[0] = img_tensor_batch.size(0);
	 inputDims.d[1] = img_tensor_batch.size(1);
	 inputDims.d[2] = img_tensor_batch.size(2);
	 inputDims.d[3] = img_tensor_batch.size(3);
	 context->setBindingDimensions(0, inputDims);
 
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
 
	 // Handle dynamic dimensions and allocate output memory
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
 
	 // Setup timing
	 vector<float> latencies;
	 cudaEvent_t start, stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
 
	 // Warm-up runs to eliminate initialization overhead
	 for (int i = 0; i < 10; ++i) {
		 context->enqueueV2(bindings.data(), stream, nullptr);
	 }
 
	 // Perform timed inference runs
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
 
	 //--------------------------------------------------
	 // Post-processing
	 //--------------------------------------------------
 
	 // Copy last output tensor back to host after all trials
	 float* last_h_output = h_outputs.back();
	 void* last_d_output = d_outputs.back();
	 int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	 cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	 cudaStreamSynchronize(stream);
 
	 // Calculate statistics for the output tensor
	 float min_val = *min_element(last_h_output, last_h_output + last_output_size);
	 float max_val = *max_element(last_h_output, last_h_output + last_output_size);
	 float avg_val = accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	 cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;
 
	 float average_latency = accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	 cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	 cudaEventDestroy(start);
	 cudaEventDestroy(stop);
 
	 // Extract output dimensions
	 int batch = outputDims.d[0];
	 int num_classes = outputDims.d[1];
	 int height = outputDims.d[2];
	 int width = outputDims.d[3];
 
	 auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);
 
	 // Debug: Print shape of the output tensor
	 std::cout << "\nNumber of dimensions(last_output_tensor): " << last_output_tensor.dim() << std::endl;
	 for (int i = 0; i < last_output_tensor.dim(); ++i) {
		 std::cout << last_output_tensor.size(i) << " ";
	 }
	 std::cout << std::endl;
 
	 // Get class predictions using argmax along class dimension
	 auto max_out = torch::max(last_output_tensor, 1);
	 auto class_labels = std::get<1>(max_out);
 
	 // Class distribution debugging (commented out for production)
	 /*
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
	 */
 
	 // Scale class indices to 8-bit range for visualization
	 int scale = 255 / 21;
	 auto scale_tensor = torch::tensor(scale, class_labels.options());
	 auto image_post = class_labels * scale;
 
	 // Process each image in the batch
	 for (int i = 0; i < batch; ++i) {
		 // Extract a single image and convert to 8-bit
		 auto single_image_post = image_post[i].squeeze().to(torch::kU8);
		 cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());
 
		 // Add each grayscale image to the result vector
		 grayscale_images.push_back(cv_img.clone());
		 
		 try {
			 // Save the segmentation output as an image
			 cv::imwrite("pngOutput/trt_seg_output_scaled_" + std::to_string(i) + ".png", cv_img);
			 cout << "Saved IMG: trt_seg_output_scaled_" + std::to_string(i) << endl;
		 }
		 catch (const cv::Exception& ex) {
			 cerr << "Failed to save image trt_seg_output_scaled_" + std::to_string(i) + " because ERROR:" << ex.what() << endl;
		 }
	 }
 
	 // Clean up resources
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
 
 /**
  * @brief Optimized segmentation inference for a single image.
  * 
  * This function provides an optimized pipeline for performing segmentation inference
  * on a single GPU tensor with minimal memory transfers and improved GPU utilization.
  * 
  * @param trt_plan Path to the TensorRT plan file
  * @param img_tensor Input image tensor (must be on CUDA device)
  * @param num_trials Number of inference trials to run
  * @return Vector containing the segmentation mask
  */
 std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_OPT(
	 const std::string& trt_plan, torch::Tensor img_tensor, int num_trials)
 {
	 std::vector<cv::Mat> grayscale_images;  // for saving the grayscale segmentation image
 
	 std::cout << "STARTING measure_segmentation_trt_performance_mul (Single Frame)" << std::endl;
	 
	 // Initialize TensorRT runtime and engine
	 TRTGeneration::CustomLogger myLogger;
	 IRuntime* runtime = createInferRuntime(myLogger);
 
	 // Read the TensorRT engine plan file (binary mode)
	 std::ifstream planFile(trt_plan, std::ios::binary);
	 std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
		 std::istreambuf_iterator<char>());
 
	 // Deserialize the engine and create execution context
	 ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	 IExecutionContext* context = engine->createExecutionContext();
	 if (!engine || !context) {
		 std::cerr << "Failed to deserialize engine or create execution context." << std::endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Verify that input tensor is on CUDA device
	 TORCH_CHECK(img_tensor.device().is_cuda(), "Input tensor must be on CUDA");
 
	 // Calculate input size
	 int input_size = img_tensor.numel();
 
	 // Create CUDA stream
	 cudaStream_t stream;
	 cudaStreamCreate(&stream);
 
	 // Allocate device memory for input
	 void* d_input;
	 cudaError_t status = cudaMalloc(&d_input, input_size * sizeof(float));
	 if (status != cudaSuccess) {
		 std::cerr << "Failed to allocate device memory for input." << std::endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Device-to-device copy directly from the GPU tensor's data pointer
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
 
	 // Create the bindings vector and add the GPU input pointer
	 std::vector<void*> bindings;
	 bindings.push_back(d_input);
 
	 // Set up output bindings
	 int numBindings = engine->getNbBindings();
	 std::vector<int> outputBindingIndices;
	 for (int i = 1; i < numBindings; ++i) {
		 outputBindingIndices.push_back(i);
	 }
 
	 // Extract dimensions from the input tensor (shape [1, C, H, W])
	 nvinfer1::Dims4 inputDims;
	 inputDims.d[0] = img_tensor.size(0);  // should be 1
	 inputDims.d[1] = img_tensor.size(1);
	 inputDims.d[2] = img_tensor.size(2);
	 inputDims.d[3] = img_tensor.size(3);
	 context->setBindingDimensions(0, inputDims);
 
	 // Allocate memory for outputs
	 std::vector<void*> d_outputs;
	 std::vector<float*> h_outputs;
	 nvinfer1::Dims outputDims;
	 for (int i : outputBindingIndices) {
		 outputDims = engine->getBindingDimensions(i);
		 // Fix dynamic dimensions if necessary
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
 
	 // Timing inference using CUDA events
	 std::vector<float> latencies;
	 cudaEvent_t start, stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start, stream);
 
	 // Run inference trials
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
 
	 // Copy output tensor back to host
	 float* last_h_output = h_outputs.back();
	 void* last_d_output = d_outputs.back();
	 int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	 cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	 cudaStreamSynchronize(stream);
 
	 // Calculate statistics for the output tensor
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
 
	 //--------------------------------------------------
	 // Post-processing
	 //--------------------------------------------------
 
	 // Extract output dimensions
	 int batch = outputDims.d[0];       // should be 1
	 int num_classes = outputDims.d[1];
	 int height = outputDims.d[2];
	 int width = outputDims.d[3];
 
	 // Create a Torch tensor from the output data
	 auto last_output_tensor = torch::from_blob(
		 last_h_output, { batch, num_classes, height, width }, torch::kFloat32);
 
	 // Get segmentation prediction via argmax along the class channel
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
 
	 // Scale class indices to 8-bit range for visualization
	 int scale = 255 / 21;
	 auto image_post = remapped_labels * scale;
 
	 // Convert the single segmentation mask to a cv::Mat
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
 
	 // Clean up resources
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
 
 //-----------------------------------------------------------------------------
 // Multi-Stream Concurrent Inference
 //-----------------------------------------------------------------------------
 
 /**
  * @brief Performs concurrent segmentation inference using multiple streams.
  * 
  * This function divides a batch of images into sub-batches and processes each 
  * sub-batch using a separate CUDA stream and execution context, maximizing 
  * throughput by running operations concurrently.
  * 
  * @param trt_plan Path to the TensorRT plan file
  * @param img_tensor_batch Batch of input images as a single tensor
  * @param num_trials Number of inference trials for warm-up
  * @return Vector of grayscale segmentation masks
  */
 std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent(
	 const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
 
	 std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent (multi-stream concurrent version)" << std::endl;
 
	 // Initialize TensorRT runtime and engine
	 TRTGeneration::CustomLogger myLogger;
	 IRuntime* runtime = createInferRuntime(myLogger);
	 ifstream planFile(trt_plan, ios::binary);
	 vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	 ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	 if (!engine) {
		 std::cerr << "Failed to deserialize engine in concurrent segmentation." << std::endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Setup multi-threading parameters
	 int totalBatch = img_tensor_batch.size(0);  // Total number of images
	 int numThreads = 2;                         // Fixed number of threads
	 int subBatch = (totalBatch + numThreads - 1) / numThreads;  // Images per thread
 
	 // Prepare result containers
	 std::vector<cv::Mat> allResults(totalBatch);
	 std::mutex resultMutex;  // Mutex to protect shared results vector
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
 
			 // Create a CUDA stream with non-blocking flags
			 cudaStream_t stream;
			 checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
 
			 // Calculate sub-batch indices for this thread
			 int startIdx = t * subBatch;
			 int endIdx = std::min(startIdx + subBatch, totalBatch);
			 int validCount = endIdx - startIdx;  // Number of valid images in this sub-batch
			 if (validCount <= 0)
				 return;
 
			 // Extract sub-batch from the global image tensor
			 torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);
			 
			 // Remove extra singleton dimension if present
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
 
			 // Allocate pinned host memory for input
			 float* h_input = nullptr;
			 checkCudaErrors(cudaMallocHost((void**)&h_input, subTensor.numel() * sizeof(float)));
			 std::memcpy(h_input, subTensor.data_ptr<float>(), subTensor.numel() * sizeof(float));
 
			 // Allocate device memory for input
			 void* d_input = nullptr;
			 checkCudaErrors(cudaMalloc(&d_input, subTensor.numel() * sizeof(float)));
			 checkCudaErrors(cudaMemcpyAsync(d_input, h_input, subTensor.numel() * sizeof(float), 
											cudaMemcpyHostToDevice, stream));
 
			 // Set input dimensions for the context
			 nvinfer1::Dims4 inputDims;
			 inputDims.d[0] = subTensor.size(0);
			 inputDims.d[1] = subTensor.size(1);
			 inputDims.d[2] = subTensor.size(2);
			 inputDims.d[3] = subTensor.size(3);
			 context->setBindingDimensions(0, inputDims);
 
			 // Prepare output bindings
			 std::vector<void*> bindings;
			 bindings.push_back(d_input);  // Binding index 0: Input buffer
			 
			 // Allocate output memory
			 int numBindings = engine->getNbBindings();
			 std::vector<float*> h_outputs;
			 std::vector<void*> d_outputs;
			 nvinfer1::Dims4 outputDims;
			 
			 for (int i = 1; i < numBindings; ++i) {
				 // Retrieve binding dimensions for each output
				 nvinfer1::Dims dims = context->getBindingDimensions(i);
				 outputDims.nbDims = dims.nbDims;
				 for (int j = 0; j < dims.nbDims; ++j) {
					 outputDims.d[j] = dims.d[j];
					 if (outputDims.d[j] < 0)
						 outputDims.d[j] = inputDims.d[j];  // Fallback to input dimensions if negative
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
 
			 // Perform warm-up runs
			 for (int i = 0; i < 3; ++i) {
				 context->enqueueV2(bindings.data(), stream, nullptr);
			 }
 
			 // Perform inference
			 if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
				 std::cerr << "TensorRT enqueueV2 failed in thread " << t << std::endl;
				 exit(EXIT_FAILURE);
			 }
 
			 // Copy results from device to host
			 int outSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
			 float* lastOutput = h_outputs.back();
			 checkCudaErrors(cudaMemcpyAsync(lastOutput, d_outputs.back(), outSize * sizeof(float), 
											cudaMemcpyDeviceToHost, stream));
			 cudaStreamSynchronize(stream);
 
			 // Post-process the output tensor
			 auto outputTensor = torch::from_blob(lastOutput, 
				 { outputDims.d[0], outputDims.d[1], outputDims.d[2], outputDims.d[3] }, torch::kFloat32);
			 
			 auto maxOut = torch::max(outputTensor, 1);  // Find max values along the channel dimension
			 auto segMask = std::get<1>(maxOut);  // Get class indices
			 
			 int scaleFactor = 255 / 21;  // Scale factor for visualization
			 auto imagePost = segMask * scaleFactor;
 
			 // Convert tensor outputs to cv::Mat and store results
			 std::vector<cv::Mat> localResults;
			 for (int i = 0; i < validCount; ++i) {
				 auto single = imagePost[i].to(torch::kU8);  // Convert to unsigned 8-bit
				 cv::Mat result(single.size(0), single.size(1), CV_8UC1, single.data_ptr<uchar>());
				 localResults.push_back(result.clone());
			 }
 
			 // Update the global results vector with thread-local results
			 {
				 std::lock_guard<std::mutex> lock(resultMutex);
				 for (int i = 0; i < validCount; ++i) {
					 allResults[startIdx + i] = localResults[i];
				 }
			 }
 
			 // Clean up resources
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
 
	 // Wait for all threads to complete
	 for (auto& t : threads) {
		 t.join();
	 }
 
	 // Clean up global resources
	 engine->destroy();
	 runtime->destroy();
 
	 return allResults;
 }
 
 /**
  * @brief Performs segmentation with CUDA Graph acceleration.
  * 
  * This function implements a hybrid approach to CUDA Graph acceleration by:
  * - Using separate streams for graph-compatible operations and memory transfers
  * - Capturing only compatible operations in CUDA Graphs
  * - Falling back to regular execution when needed
  * 
  * @param trt_plan Path to the TensorRT plan file
  * @param img_tensor_batch Batch of input images as a single tensor
  * @param num_trials Number of inference trials for warm-up
  * @return Vector of grayscale segmentation masks
  */
 std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent_graph(
	 const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
 
	 std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent_graph (Hybrid CUDA Graph approach)" << std::endl;
 
	 // Create TensorRT runtime and deserialize the engine
	 TRTGeneration::CustomLogger myLogger;
	 IRuntime* runtime = createInferRuntime(myLogger);
	 ifstream planFile(trt_plan, ios::binary);
	 vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	 std::cout << "Loaded engine size: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;
 
	 // Time engine deserialization
	 auto start_time = std::chrono::high_resolution_clock::now();
	 ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	 auto end_time = std::chrono::high_resolution_clock::now();
	 auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	 std::cout << "Deserialization required " << duration.count() << " microseconds." << std::endl;
 
	 if (!engine) {
		 std::cerr << "Failed to deserialize engine in graph segmentation." << std::endl;
		 exit(EXIT_FAILURE);
	 }
 
	 // Setup multi-threading parameters
	 int totalBatch = img_tensor_batch.size(0);
	 int numThreads = 2;
	 int subBatch = (totalBatch + numThreads - 1) / numThreads;
	 
	 // Prepare result containers
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
 
			 // Create three separate streams for different operations
			 cudaStream_t preStream, inferStream, postStream;
			 checkCudaErrors(cudaStreamCreateWithFlags(&preStream, cudaStreamNonBlocking));
			 checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			 checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));
 
			 // Calculate sub-batch indices for this thread
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
 
			 // Declare graph objects
			 cudaGraph_t postprocessGraph = nullptr;
			 cudaGraphExec_t postprocessGraphExec = nullptr;
			 bool useGraph = false;
 
			 // Declare result containers
			 std::vector<cv::Mat> localResults;
			 unsigned char* h_argmax_output = nullptr;
 
			 // First perform a regular inference run for warmup
			 for (int i = 0; i < 2; ++i) {
				 // Run TensorRT inference
				 if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
					 std::cerr << "TensorRT enqueueV2 failed during warmup" << std::endl;
					 goto cleanup; // Jump to resource cleanup
				 }
				 checkCudaErrors(cudaStreamSynchronize(inferStream));
			 }
 
			 // Try to create a graph for post-processing operations
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
 
			 // For TensorRT inference, always use regular execution
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
 
 /**
  * @brief Performs parallel inference using triple buffering technique with preloaded engine.
  * 
  * This advanced method implements triple buffering to maximize GPU utilization
  * by overlapping compute and memory operations across multiple streams. It uses
  * a pool of pre-allocated engines and contexts for maximum performance.
  * 
  * @param engine Pointer to preloaded TensorRT engine
  * @param img_tensors Vector of individual image tensors
  * @param num_streams Number of parallel streams to use
  * @return Vector of segmentation masks
  */
 std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel_preloaded_triple_buffer(
	 nvinfer1::ICudaEngine* engine, const std::vector<torch::Tensor>& img_tensors, int num_streams) {
 
	 if (!engine) {
		 std::cerr << "Error: Null engine pointer provided" << std::endl;
		 return {};
	 }
 
	 std::cout << "Starting optimized parallel inference with preloaded engine" << std::endl;
 
	 // Number of images to process
	 int num_images = img_tensors.size();
	 if (num_images == 0) {
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
			 nvinfer1::IExecutionContext* context = engine->createExecutionContext();
			 if (!context) {
				 std::cerr << "Error: Failed to create execution context for worker " << t << std::endl;
				 return;
			 }
 
			 // Create CUDA streams for this worker - one for inference and one for post-processing
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
 
				 // Verify tensor dimensions (expecting a 4D tensor with batch size 1)
				 if (img_tensor.dim() != 4 || img_tensor.size(0) != 1) {
					 std::cerr << "Error: Invalid tensor dimensions for image " << img_idx
						 << ". Expected 4D tensor with batch size 1." << std::endl;
					 continue;
				 }
 
				 try {
					 // --- PRE-ALLOCATION OF ALL MEMORY (BEFORE ANY GRAPH CAPTURE) ---
 
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
 
					 // Since the tensor is already on the GPU, we use its device pointer directly
					 size_t input_size = img_tensor.numel();
					 const float* d_input_tensor = img_tensor.data_ptr<float>();
 
					 // Set up bindings: binding index 0 is input
					 std::vector<void*> bindings;
					 bindings.push_back(const_cast<float*>(d_input_tensor));
 
					 // Allocate and bind output memory
					 std::vector<void*> d_outputs;
					 std::vector<float*> h_outputs;
					 nvinfer1::Dims outputDims;
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
 
					 // Verify that outputs were correctly allocated
					 if (h_outputs.size() != engine->getNbBindings() - 1) {
						 continue; // Skip to next image if allocation failed
					 }
 
					 // Get output dimensions (assume binding index 1 holds the needed output info)
					 int batch = outputDims.d[0]; // Should be 1
					 int num_classes = outputDims.d[1];
					 int height = outputDims.d[2];
					 int width = outputDims.d[3];
 
					 // Allocate memory for segmentation mask output (device)
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
 
					 // --- RUN TENSORRT INFERENCE (NO GRAPH CAPTURE) ---
					 cudaEvent_t start, stop;
					 cudaEventCreate(&start);
					 cudaEventCreate(&stop);
					 cudaEventRecord(start, inferStream);
 
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
 
					 // --- POST-PROCESSING WITH CUDA GRAPH (or fallback) ---
					 if (!postGraphCaptured) {
						 try {
							 cuda_error = cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed);
							 if (cuda_error != cudaSuccess) {
								 throw std::runtime_error(std::string("Failed to begin graph capture: ") +
									 cudaGetErrorString(cuda_error));
							 }
							 launchArgmaxKernel(static_cast<float*>(d_outputs.back()),
								 d_argmax_output,
								 1, // batch size is 1
								 num_classes,
								 height,
								 width,
								 postStream);
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
							 std::cerr << "Worker " << t << ": CUDA Graph capture for post-processing failed: " << e.what() << std::endl;
							 std::cerr << "Falling back to normal execution mode for post-processing" << std::endl;
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
 
					 if (postGraphCaptured && postprocessGraphExec) {
						 cuda_error = cudaGraphLaunch(postprocessGraphExec, postStream);
						 if (cuda_error != cudaSuccess) {
							 std::cerr << "Error launching post-processing graph: " << cudaGetErrorString(cuda_error) << std::endl;
							 launchArgmaxKernel(static_cast<float*>(d_outputs.back()),
								 d_argmax_output,
								 1, num_classes, height, width, postStream);
						 }
					 }
					 else {
						 launchArgmaxKernel(static_cast<float*>(d_outputs.back()),
							 d_argmax_output,
							 1, num_classes, height, width, postStream);
					 }
					 cudaStreamSynchronize(postStream);
 
					 // Allocate host buffer for segmentation mask output
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
 
					 cudaEventRecord(stop, inferStream);
					 cudaStreamSynchronize(inferStream);
					 float milliseconds = 0;
					 cudaEventElapsedTime(&milliseconds, start, stop);
 
					 // Create OpenCV Mat from segmentation mask and update results
					 cv::Mat result(height, width, CV_8UC1);
					 std::memcpy(result.data, h_argmax_output, height * width * sizeof(unsigned char));
					 {
						 std::lock_guard<std::mutex> lock(resultMutex);
						 results[img_idx] = result.clone();
					 }
					 local_frames_processed++;
 
					 // Cleanup per-image resources
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
 
			 auto worker_end_time = std::chrono::high_resolution_clock::now();
			 double total_seconds = std::chrono::duration<double>(worker_end_time - worker_start_time).count();
 
			 {
				 std::lock_guard<std::mutex> lock(resultMutex);
				 processing_times[t] = total_seconds;
				 frames_processed[t] = local_frames_processed;
			 }
 
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
 
	 for (auto& t : threads) {
		 t.join();
	 }
 
	 // Print performance summary
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
		 if (graph_usage[t])
			 graph_workers++;
	 }
 
	 std::cout << "Average processing time per worker: " << total_processing_time / num_streams << " seconds" << std::endl;
	 std::cout << "Effective overall throughput: " << num_images / (total_processing_time / num_streams) << " fps" << std::endl;
	 std::cout << "Workers using CUDA Graph: " << graph_workers << " of " << num_streams << std::endl;
	 std::cout << "============================" << std::endl;
 
	 return results;
 }
 
 //-----------------------------------------------------------------------------
 // Single-Batch Parallel Inference
 //-----------------------------------------------------------------------------
 
 /**
  * @brief Performs parallel inference using single-batch TensorRT model.
  * 
  * This function processes multiple images in parallel using a single-batch TensorRT model
  * by creating multiple execution contexts, each running on its own CUDA stream.
  * It leverages a persistent engine pool for optimal performance.
  * 
  * @param trt_plan Path to the TensorRT plan file
  * @param img_tensors Vector of individual image tensors
  * @param num_streams Number of parallel streams to use
  * @return Vector of segmentation masks
  */
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel_preloaded(
    const std::string& trt_plan,
    const std::vector<torch::Tensor>& img_tensors,
    int num_streams)
{
    // Force a batch size of 2 frames
    const int BATCH_SIZE = 2;
    int num_images = std::min((int)img_tensors.size(), BATCH_SIZE);
    if (num_images == 0) {
        return {};
    }

    std::cout << "Starting optimized parallel single-batch segmentation on "
        << num_images << " frames (batch size = " << BATCH_SIZE << ")"
        << " with persistent engine pool, persistent output buffers, and persistent CUDA streams/events." << std::endl;

    // Build or reuse a persistent pool of engine-context pairs
    {
        std::lock_guard<std::mutex> lock(enginePoolMutex);
        if (persistentEnginePool.size() < static_cast<size_t>(num_streams)) {
            int enginesToCreate = num_streams - persistentEnginePool.size();
            std::ifstream planFile(trt_plan, std::ios::binary);
            if (!planFile.is_open()) {
                std::cerr << "Error: Could not open plan file: " << trt_plan << std::endl;
                return {};
            }
            std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
                std::istreambuf_iterator<char>());
            for (int i = 0; i < enginesToCreate; ++i) {
                TRTGeneration::CustomLogger myLogger;
                nvinfer1::IRuntime* runtime = createInferRuntime(myLogger);
                if (!runtime) {
                    std::cerr << "Error: Failed to create runtime." << std::endl;
                    continue;
                }
                nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
                if (!engine) {
                    std::cerr << "Error: Failed to deserialize engine." << std::endl;
                    runtime->destroy();
                    continue;
                }
                nvinfer1::IExecutionContext* context = engine->createExecutionContext();
                if (!context) {
                    std::cerr << "Error: Failed to create execution context." << std::endl;
                    engine->destroy();
                    runtime->destroy();
                    continue;
                }
                EngineContextPair pair;
                pair.engine = engine;
                pair.context = context;
                // Initially, buffers and streams/events are not allocated
                persistentEnginePool.push_back(pair);
            }
            if (persistentEnginePool.size() < static_cast<size_t>(num_streams)) {
                std::cerr << "Error: Insufficient engine-context pairs were created." << std::endl;
                return {};
            }
        }
    }

    // For this call, assign one engine-context pair per worker thread
    std::vector<EngineContextPair*> contextPool;
    {
        std::lock_guard<std::mutex> lock(enginePoolMutex);
        for (int i = 0; i < num_streams; ++i)
            contextPool.push_back(&persistentEnginePool[i]);
    }

    // Prepare results and threading parameters
    std::vector<cv::Mat> results(num_images);
    std::mutex resultMutex;
    std::vector<std::thread> threads;
    int images_per_thread = (BATCH_SIZE + num_streams - 1) / num_streams;
    std::vector<double> processing_times(num_streams, 0.0);
    std::vector<int> frames_processed(num_streams, 0);
    std::vector<bool> graph_usage(num_streams, false);

    // Launch worker threads
    for (int t = 0; t < num_streams; ++t) {
        threads.emplace_back([&, t]() {
            // Determine the image range for this worker, capped by BATCH_SIZE
            int start_idx = t * images_per_thread;
            int end_idx = std::min(start_idx + images_per_thread, BATCH_SIZE);
            if (start_idx >= num_images)
                return;

            // Retrieve the engine-context pair for this worker
            EngineContextPair* pair = contextPool[t];
            nvinfer1::IExecutionContext* context = pair->context;
            nvinfer1::ICudaEngine* engine = pair->engine;
            if (!context) {
                std::cerr << "Error: No context available for worker " << t << std::endl;
                return;
            }

            // Reuse persistent CUDA streams and events if available; otherwise, create them
            if (!pair->streamsEventsInitialized) {
                checkCudaErrors(cudaStreamCreateWithFlags(&pair->inferStream, cudaStreamNonBlocking));
                checkCudaErrors(cudaStreamCreateWithFlags(&pair->postStream, cudaStreamNonBlocking));
                checkCudaErrors(cudaEventCreate(&pair->startEvent));
                checkCudaErrors(cudaEventCreate(&pair->stopEvent));
                pair->streamsEventsInitialized = true;
            }
            // Use persistent streams/events
            cudaStream_t inferStream = pair->inferStream;
            cudaStream_t postStream = pair->postStream;
            cudaEvent_t startEvent = pair->startEvent;
            cudaEvent_t stopEvent = pair->stopEvent;

            // Persistent output buffers are handled as before
            if (!pair->buffersAllocated) {
                for (int i = 1; i < engine->getNbBindings(); ++i) {
                    nvinfer1::Dims outDims = context->getBindingDimensions(i);
                    int outputSize = 1;
                    for (int j = 0; j < outDims.nbDims; ++j) {
                        outputSize *= outDims.d[j];
                    }
                    float* h_output = nullptr;
                    void* d_output = nullptr;
                    cudaError_t cuda_error = cudaMallocHost((void**)&h_output, outputSize * sizeof(float));
                    if (cuda_error != cudaSuccess) {
                        std::cerr << "Error allocating persistent host output memory: " << cudaGetErrorString(cuda_error) << std::endl;
                        break;
                    }
                    cuda_error = cudaMalloc(&d_output, outputSize * sizeof(float));
                    if (cuda_error != cudaSuccess) {
                        std::cerr << "Error allocating persistent device output memory: " << cudaGetErrorString(cuda_error) << std::endl;
                        cudaFreeHost(h_output);
                        break;
                    }
                    pair->persistent_h_outputs.push_back(h_output);
                    pair->persistent_d_outputs.push_back(d_output);
                    pair->persistent_output_dims.push_back(outDims);
                }
                if (pair->persistent_h_outputs.size() == static_cast<size_t>(engine->getNbBindings() - 1))
                    pair->buffersAllocated = true;
                else {
                    std::cerr << "Error: Persistent buffer allocation failed for worker " << t << std::endl;
                }
            }

            // Helper lambda to build bindings using persistent buffers
            auto buildBindings = [&]() -> std::vector<void*> {
                std::vector<void*> bindings;
                bindings.push_back(nullptr); // binding 0 for input (set per image)
                for (int i = 0; i < pair->persistent_d_outputs.size(); ++i) {
                    bindings.push_back(pair->persistent_d_outputs[i]);
                }
                return bindings;
            };

            auto worker_start_time = std::chrono::high_resolution_clock::now();
            int local_frames_processed = 0;

            // Variables for CUDA graph post-processing
            cudaGraph_t postprocessGraph = nullptr;
            cudaGraphExec_t postprocessGraphExec = nullptr;
            bool postGraphCaptured = false;

            for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
                const torch::Tensor& img_tensor = img_tensors[img_idx];
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
                    // Set input dimensions
                    nvinfer1::Dims4 inputDims;
                    inputDims.d[0] = 1;
                    inputDims.d[1] = img_tensor.size(1);
                    inputDims.d[2] = img_tensor.size(2);
                    inputDims.d[3] = img_tensor.size(3);
                    context->setBindingDimensions(0, inputDims);
                    if (!context->allInputDimensionsSpecified()) {
                        std::cerr << "Error: Not all input dimensions specified for image " << img_idx << std::endl;
                        continue;
                    }

                    // Get input device pointer
                    float* d_input = const_cast<float*>(img_tensor.data_ptr<float>());
                    std::vector<void*> bindings = buildBindings();
                    bindings[0] = d_input; // Set input binding

                    // Record start event using persistent event
                    checkCudaErrors(cudaEventRecord(startEvent, inferStream));

                    // Run inference
                    if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
                        std::cerr << "Error: TensorRT enqueueV2 failed for image " << img_idx << std::endl;
                        continue;
                    }
                    cudaStreamSynchronize(inferStream);

                    // Record stop event and compute elapsed time
                    checkCudaErrors(cudaEventRecord(stopEvent, inferStream));
                    cudaStreamSynchronize(inferStream);
                    float milliseconds = 0;
                    checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

                    // Use persistent output dimensions (assume binding[1] holds segmentation)
                    int num_classes = pair->persistent_output_dims[0].d[1];
                    int height = pair->persistent_output_dims[0].d[2];
                    int width = pair->persistent_output_dims[0].d[3];

                    // Allocate temporary device buffer for argmax output
                    unsigned char* d_argmax_output = nullptr;
                    cudaError_t cuda_error = cudaMalloc(&d_argmax_output, height * width * sizeof(unsigned char));
                    if (cuda_error != cudaSuccess) {
                        std::cerr << "Error allocating argmax output memory: " << cudaGetErrorString(cuda_error) << std::endl;
                        continue;
                    }

                    // Capture or launch CUDA graph for post-processing
                    if (!postGraphCaptured) {
                        try {
                            cuda_error = cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed);
                            if (cuda_error != cudaSuccess) {
                                throw std::runtime_error(std::string("Failed to begin graph capture: ") +
                                    cudaGetErrorString(cuda_error));
                            }
                            launchArgmaxKernel(
                                static_cast<float*>(pair->persistent_d_outputs.back()),
                                d_argmax_output,
                                1, // batch size is 1
                                num_classes,
                                height,
                                width,
                                postStream
                            );
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
                            std::cout << "Worker " << t << ": Successfully captured post-processing graph" << std::endl;
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Worker " << t << ": CUDA Graph capture failed: " << e.what() << std::endl;
                            if (postprocessGraph) { cudaGraphDestroy(postprocessGraph); postprocessGraph = nullptr; }
                            if (postprocessGraphExec) { cudaGraphExecDestroy(postprocessGraphExec); postprocessGraphExec = nullptr; }
                        }
                    }

                    if (postGraphCaptured && postprocessGraphExec) {
                        cuda_error = cudaGraphLaunch(postprocessGraphExec, postStream);
                        if (cuda_error != cudaSuccess) {
                            std::cerr << "Error launching post-processing graph: " << cudaGetErrorString(cuda_error) << std::endl;
                            launchArgmaxKernel(
                                static_cast<float*>(pair->persistent_d_outputs.back()),
                                d_argmax_output,
                                1,
                                num_classes,
                                height,
                                width,
                                postStream
                            );
                        }
                    }
                    else {
                        launchArgmaxKernel(
                            static_cast<float*>(pair->persistent_d_outputs.back()),
                            d_argmax_output,
                            1,
                            num_classes,
                            height,
                            width,
                            postStream
                        );
                    }
                    cudaStreamSynchronize(postStream);

                    // Allocate temporary host buffer for argmax result
                    unsigned char* h_argmax_output = new unsigned char[height * width];
                    cuda_error = cudaMemcpyAsync(h_argmax_output, d_argmax_output,
                        height * width * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost, postStream);
                    if (cuda_error != cudaSuccess) {
                        std::cerr << "Error copying results to host: " << cudaGetErrorString(cuda_error) << std::endl;
                        delete[] h_argmax_output;
                        cudaFree(d_argmax_output);
                        continue;
                    }
                    cudaStreamSynchronize(postStream);

                    // Create an OpenCV Mat from the result
                    cv::Mat result(height, width, CV_8UC1);
                    std::memcpy(result.data, h_argmax_output, height * width * sizeof(unsigned char));
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        results[img_idx] = result.clone();
                    }
                    local_frames_processed++;

                    delete[] h_argmax_output;
                    cudaFree(d_argmax_output);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing image " << img_idx << ": " << e.what() << std::endl;
                }
            } // End per-image loop

            auto worker_end_time = std::chrono::high_resolution_clock::now();
            double total_seconds = std::chrono::duration<double>(worker_end_time - worker_start_time).count();
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                processing_times[t] = total_seconds;
                frames_processed[t] = local_frames_processed;
            }
            if (postprocessGraphExec) { cudaGraphExecDestroy(postprocessGraphExec); }
            if (postprocessGraph) { cudaGraphDestroy(postprocessGraph); }
            // Note: Persistent streams/events are retained for reuse
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return results;
}

//-----------------------------------------------------------------------------
// Super-Resolution Inference
//-----------------------------------------------------------------------------

/**
 * @brief Measures performance of super-resolution inference.
 * 
 * This function loads a TensorRT plan file, performs inference on a single input tensor,
 * and optionally compares the output with an original image for quality assessment.
 * 
 * @param trt_plan Path to the serialized TensorRT engine plan file
 * @param original_image_path Path to the original high-resolution image for comparison
 * @param img_tensor Input tensor in NCHW format (low-resolution image)
 * @param num_trials Number of inference runs for performance measurement
 * @param compare_img_bool Whether to compare output with original image
 */
void TRTInference::measure_trt_performance(const string& trt_plan,
    const string& original_image_path,
    torch::Tensor img_tensor,
    int num_trials,
    bool compare_img_bool) {

    std::cout << "STARTING measure_trt_performance" << std::endl;

    // Initialize TensorRT runtime and engine
    TRTGeneration::CustomLogger myLogger;
    IRuntime* runtime = createInferRuntime(myLogger);

    // Read the TensorRT engine plan file (binary mode)
    ifstream planFile(trt_plan, ios::binary);
    vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

    // Deserialize the engine and create execution context
    ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    IExecutionContext* context = engine->createExecutionContext();
    if (!engine || !context) {
        cerr << "Failed to deserialize engine or create execution context." << endl;
        exit(EXIT_FAILURE);
    }

    // Allocate pinned memory for input data
    int input_size = img_tensor.numel();
    float* h_input;
    cudaMallocHost((void**)&h_input, input_size * sizeof(float));

    // Set binding dimensions and identify output bindings
    nvinfer1::Dims4 inputDims;
    nvinfer1::Dims4 outputDims;

    std::vector<int> outputBindingIndices;
    std::vector<std::string> outputTensorNames;
    for (int i = 1; i < engine->getNbBindings(); ++i) {
        outputBindingIndices.push_back(i);
        outputTensorNames.push_back(engine->getBindingName(i));
    }

    // Extract dimensions from input tensor (NCHW format)
    inputDims.d[0] = img_tensor.size(0);
    inputDims.d[1] = img_tensor.size(1);
    inputDims.d[2] = img_tensor.size(2);
    inputDims.d[3] = img_tensor.size(3);
    context->setBindingDimensions(0, inputDims);

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
    
    // Transfer input data to GPU
    cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (memcpyStatus != cudaSuccess) {
        cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
        exit(EXIT_FAILURE);
    }
    bindings.push_back(d_input);

    // Allocate memory for outputs
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

    // Setup timing
    vector<float> latencies;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        context->enqueueV2(bindings.data(), stream, nullptr);
    }

    // Perform timed inference runs
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

    // Copy output tensor back to host
    float* last_h_output = h_outputs.back();
    int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
    cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate statistics for the output tensor
    float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
    float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
    float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
    cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

    float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
    cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Convert output tensor to OpenCV Mat
    cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);
    cv::Mat clipped_image_data;
    cv::min(image_data, 1.0, clipped_image_data);
    cv::max(clipped_image_data, 0.0, clipped_image_data);
    clipped_image_data *= 255;
    clipped_image_data.convertTo(clipped_image_data, CV_8U);

    // Clean up resources
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
 * @brief Performs parallel inference using individual image tensors.
 * 
 * This function processes multiple images independently using separate execution contexts
 * and CUDA streams for maximum parallelism. It's useful for cases where batch processing
 * isn't feasible or efficient.
 * 
 * @param trt_plan Path to the TensorRT plan file
 * @param img_tensors Vector of individual image tensors
 * @param num_streams Number of parallel streams to use
 * @return Vector of segmentation masks
 */
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel(
    const std::string& trt_plan, 
    const std::vector<torch::Tensor>& img_tensors, 
    int num_streams) {

    std::cout << "STARTING measure_segmentation_trt_performance_single_batch_parallel" << std::endl;
    
    // Initialize TensorRT runtime and engine
    TRTGeneration::CustomLogger myLogger;
    IRuntime* runtime = createInferRuntime(myLogger);
    
    // Read the TensorRT engine plan file (binary mode)
    std::ifstream planFile(trt_plan, std::ios::binary);
    std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
        std::istreambuf_iterator<char>());
        
    // Deserialize the engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    if (!engine) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        runtime->destroy();
        return {};
    }
    
    int num_images = img_tensors.size();
    if (num_images == 0) {
        engine->destroy();
        runtime->destroy();
        return {};
    }
    
    // Adjust number of streams based on available images
    num_streams = std::min(num_streams, num_images);
    std::cout << "Processing " << num_images << " images with " << num_streams << " parallel streams" << std::endl;
    
    // Calculate images per thread
    int images_per_thread = (num_images + num_streams - 1) / num_streams;
    
    // Prepare results and threading variables
    std::vector<cv::Mat> results(num_images);
    std::mutex resultMutex;
    std::vector<std::thread> threads;
    
    // Launch worker threads
    for (int t = 0; t < num_streams; ++t) {
        threads.emplace_back([&, t]() {
            // Calculate the range of images for this thread
            int start_idx = t * images_per_thread;
            int end_idx = std::min(start_idx + images_per_thread, num_images);
            if (start_idx >= num_images) {
                return; // No images to process
            }
            
            // Create execution context for this thread
            IExecutionContext* context = engine->createExecutionContext();
            if (!context) {
                std::cerr << "Failed to create execution context for thread " << t << std::endl;
                return;
            }
            
            // Create dedicated CUDA streams for this thread
            cudaStream_t inferStream, postStream;
            cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking);
            
            // Process each image in the assigned range
            for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
                const torch::Tensor& img_tensor = img_tensors[img_idx];
                
                // Verify that the tensor is on CUDA and has the expected dimensions
                if (!img_tensor.is_cuda() || img_tensor.dim() != 4 || img_tensor.size(0) != 1) {
                    std::cerr << "Invalid tensor for image " << img_idx << std::endl;
                    continue;
                }
                
                try {
                    // Set input dimensions
                    nvinfer1::Dims4 inputDims;
                    inputDims.d[0] = 1; // Batch size 1
                    inputDims.d[1] = img_tensor.size(1);
                    inputDims.d[2] = img_tensor.size(2);
                    inputDims.d[3] = img_tensor.size(3);
                    context->setBindingDimensions(0, inputDims);
                    
                    if (!context->allInputDimensionsSpecified()) {
                        std::cerr << "Not all input dimensions specified for image " << img_idx << std::endl;
                        continue;
                    }
                    
                    // Setup bindings
                    std::vector<void*> bindings;
                    bindings.push_back(const_cast<float*>(img_tensor.data_ptr<float>()));
                    
                    // Allocate and bind output memory
                    std::vector<void*> d_outputs;
                    std::vector<float*> h_outputs;
                    nvinfer1::Dims outputDims;
                    
                    for (int i = 1; i < engine->getNbBindings(); ++i) {
                        outputDims = context->getBindingDimensions(i);
                        int outputSize = 1;
                        for (int j = 0; j < outputDims.nbDims; ++j) {
                            outputSize *= outputDims.d[j];
                        }
                        
                        float* h_output;
                        void* d_output;
                        cudaMallocHost((void**)&h_output, outputSize * sizeof(float));
                        cudaMalloc(&d_output, outputSize * sizeof(float));
                        
                        h_outputs.push_back(h_output);
                        d_outputs.push_back(d_output);
                        bindings.push_back(d_output);
                    }
                    
                    // Run inference
                    if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
                        std::cerr << "TensorRT enqueueV2 failed for image " << img_idx << std::endl;
                        continue;
                    }
                    cudaStreamSynchronize(inferStream);
                    
                    // Get the output dimensions
                    int batch = outputDims.d[0]; // Should be 1
                    int num_classes = outputDims.d[1];
                    int height = outputDims.d[2];
                    int width = outputDims.d[3];
                    
                    // Copy output back to host
                    float* last_h_output = h_outputs.back();
                    void* last_d_output = d_outputs.back();
                    int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
                    cudaMemcpyAsync(last_h_output, last_d_output, last_output_size * sizeof(float),
                        cudaMemcpyDeviceToHost, inferStream);
                    cudaStreamSynchronize(inferStream);
                    
                    // Process the output tensor
                    auto output_tensor = torch::from_blob(
                        last_h_output, {batch, num_classes, height, width}, torch::kFloat32);
                    
                    // Get segmentation prediction via argmax along the class channel
                    auto max_out = torch::max(output_tensor, 1);
                    auto class_labels = std::get<1>(max_out);
                    
                    // Scale class indices to 8-bit range
                    int scale = 255 / 21;
                    auto image_post = class_labels * scale;
                    
                    // Convert to OpenCV Mat
                    auto single_image_post = image_post[0].squeeze().to(torch::kU8);
                    cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1,
                        single_image_post.data_ptr<uchar>());
                        
                    // Store result
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        results[img_idx] = cv_img.clone();
                    }
                    
                    // Clean up resources for this image
                    for (auto ptr : h_outputs) {
                        cudaFreeHost(ptr);
                    }
                    for (auto ptr : d_outputs) {
                        cudaFree(ptr);
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing image " << img_idx << ": " << e.what() << std::endl;
                }
            }
            
            // Clean up thread resources
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
    
    return results;
}