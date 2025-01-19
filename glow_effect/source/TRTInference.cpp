/**
* you probably need to change the output path labeled 
*/

#include "TRTInference.hpp"
#include "ImageProcessingUtil.hpp" 
#include "nvToolsExt.h"


// Single Image Segmantion Inference
void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {

    std::cout << "STARTING measure_trt_performance" << std::endl;
    TRTGeneration::CustomLogger myLogger; 
    IRuntime* runtime = createInferRuntime(myLogger);

    // reads file in binary w/o preprocessing
    ifstream planFile(trt_plan, ios::binary);
    vector<char> plan((istreambuf_iterator<char>(planFile)),istreambuf_iterator<char>());
    
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
    
    auto last_output_tensor = torch::from_blob(last_h_output, {batch, num_classes, height, width}, torch::kFloat32);

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
    auto permuted_img = image_post.permute({1, 2, 0}).to(torch::kU8);

    std::cout << "Number of dimensions(permuted_img): " << permuted_img.dim() << std::endl;
    for (int i = 0; i < permuted_img.dim(); ++i) {
        std::cout << permuted_img.size(i) << " ";
    }
 
    cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());

    try {
        ///////////////////////////////////////////////////////////////////
        cv::imwrite("pngOutput/trt_seg_output_scaled.png", cv_img);        // use your own path
        /// //////////////////////////////////////////////////////////////

        cout << "Saved IMG: trt_seg_output_scaled" << endl; 
    } catch (const cv::Exception& ex) {
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






// Super Resolution Inference
void TRTInference::measure_trt_performance(const string& trt_plan, const string& original_image_path, torch::Tensor img_tensor, int num_trials, bool compare_img_bool) {

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
    // cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

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

        // PLEASE CHANGE ACCORDING TO THE MODEL USED: Currently setting output dims to 4x as SR models output 4x the og image
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

    vector<float> latencies;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
    for (int i = 0; i < 10; ++i) {
        context->enqueueV2(bindings.data(), stream, nullptr);
    }

    cudaEventRecord(start, stream);

    for (int i = 0; i < num_trials; ++i) {
        // Run asynchronous inference using enqueueV2
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

    // Saving the last output tensor as an image
    cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);

    // Clip the values between 0 and 1
    cv::Mat clipped_image_data;
    cv::min(image_data, 1.0, clipped_image_data);
    cv::max(clipped_image_data, 0.0, clipped_image_data);

    // Multiply by 255
    clipped_image_data *= 255;

    // Convert to 8-bit image
    clipped_image_data.convertTo(clipped_image_data, CV_8U);


    try {
        cv::imwrite("./pngOutput/trt_output.png", clipped_image_data);
        cout << "Saved IMG: trt_output" << endl;
        
        cv::Mat original_image = cv::imread(original_image_path);
        if (original_image.empty()) {
            cerr << "Error: Original image not found or unable to read." << endl;
            return;
        }

        cv::Mat grayscale_original;
        cv::cvtColor(original_image, grayscale_original, cv::COLOR_BGR2GRAY);

        if (compare_img_bool == true){
            ImageProcessingUtil::compareImages(clipped_image_data, grayscale_original);
        }

    } catch (const cv::Exception& ex) {
        cerr << "Failed to save the image: " << ex.what() << endl;
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