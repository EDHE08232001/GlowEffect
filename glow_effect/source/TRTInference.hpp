#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/quality.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <future>       // for std::async, std::future
#include <thread>       // for std::thread
#include <mutex>        // for std::mutex

// Using declarations for brevity.
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

#include "TRTGeneration.hpp"
#include "ImageProcessingUtil.hpp"

/**
 * @brief A class providing TensorRT inference routines for segmentation and super-resolution.
 * 
 * This class implements various optimized inference methods using TensorRT for deep learning models,
 * with a focus on segmentation and super-resolution tasks. It provides functionality for both 
 * single-image and batched inference, along with performance measurement utilities.
 * 
 * The class supports several optimization techniques:
 * - Persistent engine loading to avoid repeated deserialization
 * - CUDA graph acceleration for repetitive operations
 * - Multiple CUDA streams for concurrent execution
 * - Triple buffering for improved GPU utilization
 * - Pinned memory for efficient host-device transfers
 */
class TRTInference {
public:
    /**
     * @brief Initializes the TensorRT engine from a serialized plan file.
     *
     * This function loads a TensorRT engine from a serialized plan file and 
     * creates a persistent execution context for reuse across multiple inference calls.
     *
     * @param trt_plan Path to the serialized TensorRT engine plan file.
     * @return True if initialization was successful, false otherwise.
     */
    static bool initializeTRTEngine(const std::string& trt_plan);

    /**
     * @brief Performs segmentation inference using a pre-loaded TensorRT engine.
     *
     * This method uses the persistent TensorRT engine and CUDA graph acceleration
     * to efficiently perform inference on a single input tensor.
     *
     * @param img_tensor Input tensor in NCHW format (must be on CUDA device).
     * @param num_trials Number of inference runs (default: 1).
     * @return Vector of segmentation masks as OpenCV Mat objects.
     */
    static std::vector<cv::Mat> performSegmentationInference(torch::Tensor img_tensor, int num_trials = 1);

    /**
     * @brief Releases resources associated with the persistent TensorRT engine.
     *
     * Frees all allocated CUDA memory, destroys the CUDA graph, execution context,
     * and TensorRT engine to prevent memory leaks.
     */
    static void cleanupTRTEngine();

    /**
     * @brief Performs parallel inference using multiple streams with preloaded engine.
     *
     * This method uses a preloaded TensorRT engine and creates multiple CUDA streams
     * to process multiple images in parallel.
     *
     * @param trt_plan Path to the serialized TensorRT engine plan file.
     * @param img_tensors Vector of individual image tensors (each with batch_size=1).
     * @param num_streams Number of parallel streams to use.
     * @return Vector of segmentation mask images.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_single_batch_parallel_preloaded(
        const std::string& trt_plan, 
        const std::vector<torch::Tensor>& img_tensors, 
        int num_streams);

    /**
     * @brief Performs parallel inference using triple buffering with preloaded engine.
     *
     * This advanced method implements triple buffering to maximize GPU utilization
     * by overlapping compute and memory operations across multiple streams.
     *
     * @param engine Pointer to preloaded TensorRT engine.
     * @param img_tensors Vector of individual image tensors (each with batch_size=1).
     * @param num_streams Number of parallel streams to use.
     * @return Vector of segmentation mask images.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_single_batch_parallel_preloaded_triple_buffer(
        nvinfer1::ICudaEngine* engine, 
        const std::vector<torch::Tensor>& img_tensors, 
        int num_streams);

    /**
     * @brief Measures the performance of TRT inference on super-resolution models.
     *
     * Loads a TensorRT plan file, performs inference on a single input tensor, and (optionally)
     * compares the output with an original image.
     *
     * @param trt_plan             Path to the serialized TensorRT engine plan file.
     * @param original_image_path  Path to the original image file for comparison.
     * @param img_tensor           A 4D tensor (NCHW) containing the preprocessed input image data.
     * @param num_trials           Number of inference runs for performance measurement.
     * @param compare_img_bool     If true, compares the final output with the original image.
     */
    static void measure_trt_performance(
        const std::string& trt_plan, 
        const std::string& original_image_path,
        torch::Tensor img_tensor, 
        int num_trials, 
        bool compare_img_bool);

    /**
     * @brief Optimized measurement of segmentation performance for a single image.
     *
     * This method provides an optimized pipeline for segmentation inference on a single image,
     * focusing on minimizing memory transfers and improving GPU utilization.
     *
     * @param trt_plan Path to the serialized TensorRT engine plan file.
     * @param img_tensor Input tensor in NCHW format (must be on CUDA device).
     * @param num_trials Number of inference runs for performance measurement.
     * @return Vector of segmentation masks as OpenCV Mat objects.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_mul_OPT(
        const std::string& trt_plan, 
        torch::Tensor img_tensor, 
        int num_trials);

    /**
     * @brief Measures the performance of TRT inference on segmentation models.
     *
     * Performs inference on a single input tensor and measures latency over a specified number of trials.
     *
     * @param trt_plan   Path to the serialized TensorRT engine plan file.
     * @param img_tensor A 4D tensor (NCHW) containing the preprocessed input image data.
     * @param num_trials Number of inference runs for performance measurement.
     */
    static void measure_segmentation_trt_performance(
        const std::string& trt_plan, 
        torch::Tensor img_tensor, 
        int num_trials);

    /**
     * @brief Performs segmentation inference on a batch of images and returns grayscale outputs.
     *
     * Loads a TensorRT plan file, processes a batched input tensor, measures latency, and converts
     * the final output into a vector of single-channel OpenCV Mats.
     *
     * @param trt_plan   Path to the serialized TensorRT engine plan file.
     * @param img_tensor A 4D tensor (NCHW) containing batched preprocessed images.
     * @param num_trials Number of inference runs for performance measurement.
     * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_mul(
        const std::string& trt_plan, 
        torch::Tensor img_tensor, 
        int num_trials);

    /**
     * @brief Performs segmentation inference on a batch of images concurrently using multiple streams.
     *
     * This function splits the input batch into sub-batches and processes each sub-batch on its own
     * non-blocking CUDA stream and execution context. Pinned memory is used for fast host-device transfers.
     * The segmentation results from all sub-batches are merged and returned as a vector of OpenCV Mats.
     *
     * @param trt_plan         Path to the serialized TensorRT engine plan file.
     * @param img_tensor_batch A 4D tensor (NCHW) containing batched preprocessed images.
     * @param num_trials       Number of inference runs for warm-up.
     * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_mul_concurrent(
        const std::string& trt_plan, 
        torch::Tensor img_tensor_batch, 
        int num_trials);

    /**
     * @brief Performs segmentation inference on a batch of images with CUDA Graph acceleration where possible.
     *
     * This function implements a hybrid approach to CUDA Graph acceleration:
     * - Uses separate streams for graph-compatible operations and memory transfers
     * - Attempts to capture only TensorRT inference in the CUDA Graph (not memory transfers)
     * - Falls back to regular execution if TensorRT's internal operations are incompatible with graph capture
     * - Reports performance metrics for either execution mode
     *
     * The function maintains the same multi-threading model as the concurrent version,
     * processing sub-batches in parallel while using pinned memory for efficient transfers.
     *
     * @param trt_plan         Path to the serialized TensorRT engine plan file.
     * @param img_tensor_batch A 4D tensor (NCHW) containing batched preprocessed images.
     * @param num_trials       Number of inference runs for warm-up.
     * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_mul_concurrent_graph(
        const std::string& trt_plan, 
        torch::Tensor img_tensor_batch, 
        int num_trials);

    /**
     * @brief Processes multiple images in parallel using a single-batch TRT model
     *
     * This implementation maximizes parallelism for single-batch TensorRT models by:
     * 1. Creating multiple execution contexts and CUDA streams (one per worker thread)
     * 2. Processing images independently and concurrently
     * 3. Using CUDA Graphs for post-processing operations only (argmax kernel)
     * 4. Properly separating inference streams from post-processing streams
     * 5. Implementing robust error handling and recovery mechanisms
     *
     * @param trt_plan Path to the single-batch TensorRT plan file
     * @param img_tensors Vector of individual image tensors (each with batch_size=1)
     * @param num_streams Number of parallel streams to use
     * @return Vector of segmentation mask images
     */
    static std::vector<cv::Mat> measure_segmentation_trt_performance_single_batch_parallel(
        const std::string& trt_plan, 
        const std::vector<torch::Tensor>& img_tensors, 
        int num_streams);

private:
    // -------------- TensorRT Engine Resources --------------
    /// Persistent TensorRT runtime for engine deserialization
    static IRuntime* s_runtime;
    /// Persistent TensorRT engine for inference
    static ICudaEngine* s_engine;
    /// Persistent execution context for the TensorRT engine
    static IExecutionContext* s_context;
    /// Flag indicating if the engine has been initialized
    static bool s_initialized;

    // -------------- CUDA Graph Resources --------------
    /// CUDA graph for accelerating repetitive operations
    static cudaGraph_t s_graph;
    /// Executable instance of the CUDA graph
    static cudaGraphExec_t s_graphExec;
    /// Flag indicating if the CUDA graph has been initialized
    static bool s_graphInitialized;
    /// Device memory for persistent input buffer
    static void* s_d_input;
    /// Device memory for persistent output buffers
    static std::vector<void*> s_d_outputs;
    /// Host memory for persistent output buffers
    static std::vector<float*> s_h_outputs;
    /// Persistent bindings vector for TensorRT execution
    static std::vector<void*> s_bindings;
    /// Persistent CUDA stream for graph execution
    static cudaStream_t s_stream;
    /// Output dimensions for persistent memory allocation
    static nvinfer1::Dims s_outputDims;

    /**
     * @brief Initializes a CUDA Graph for accelerated inference.
     *
     * This method creates and captures a CUDA graph for the inference operation,
     * which can significantly improve performance for repeated inference on inputs
     * of the same shape.
     *
     * @param sample_tensor A sample input tensor used to set up the graph.
     * @return True if graph initialization was successful, false otherwise.
     */
    static bool initializeCudaGraph(const torch::Tensor& sample_tensor);
};

#endif // TRT_INFERENCE_HPP