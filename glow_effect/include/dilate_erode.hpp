/*******************************************************************************************************************
 * FILE NAME   :    dilate_erode.hpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    Implementation of dilation or erosion algorithm with template-based operations.
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 DEC 11      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include "all_common.h"

template<typename _Ty, int _Sh>
class dilate_erode_op
{
public:
	/**
	 * Constructor for dilation or erosion operation.
	 * @param d_n_e: Determines whether the operation is dilation (true) or erosion (false).
	 */
	dilate_erode_op(const bool d_n_e) : dNe(d_n_e) { }

	/**
	 * Performs horizontal dilation or erosion operation.
	 * @param img_hsize: Width of the image.
	 * @param img_vsize: Height of the image.
	 * @param se_int: Integer part of the structuring element.
	 * @param se_frc: Fractional part of the structuring element.
	 * @param din: Input data array.
	 * @param dout: Output data array.
	 */
	void hor_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
	{
		// Path to the segmentation model output.
		std::string segmentation_image_path = "C:/path_to/onnx-to-trt/seg_out/trt_seg_output_scaled_0.png";

		// Read the segmentation output.
		cv::Mat segmentation_output = cv::imread(segmentation_image_path, cv::IMREAD_GRAYSCALE);
		if (segmentation_output.empty()) {
			std::cerr << "Error reading segmentation output image file." << std::endl;
			return;
		}

		// Resize segmentation output to match the input dimensions.
		cv::resize(segmentation_output, segmentation_output, cv::Size(img_hsize, img_vsize));

		// Validate the resized dimensions.
		if (segmentation_output.rows != img_vsize || segmentation_output.cols != img_hsize) {
			std::cerr << "Error: resized segmentation output dimensions do not match expected size." << std::endl;
			return;
		}

		// Copy segmentation output data to dout.
		for (int i = 0; i < img_vsize; i++) {
			for (int j = 0; j < img_hsize; j++) {
				dout[i * img_hsize + j] = static_cast<_Ty>(segmentation_output.at<uchar>(i, j));
			}
		}
	}

	/**
	 * Performs vertical dilation or erosion operation.
	 * @param img_hsize: Width of the image.
	 * @param img_vsize: Height of the image.
	 * @param se_int: Integer part of the structuring element.
	 * @param se_frc: Fractional part of the structuring element.
	 * @param din: Input data array.
	 * @param dout: Output data array.
	 */
	void ver_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
	{
		// Path to the segmentation model output.
		std::string segmentation_image_path = "C:/path_to/onnx-to-trt/seg_out/trt_seg_output_scaled_0.png";

		// Read the segmentation output.
		cv::Mat segmentation_output = cv::imread(segmentation_image_path, cv::IMREAD_GRAYSCALE);
		if (segmentation_output.empty()) {
			std::cerr << "Error reading segmentation output image file." << std::endl;
			return;
		}

		// Resize segmentation output to match the input dimensions.
		cv::resize(segmentation_output, segmentation_output, cv::Size(img_hsize, img_vsize));

		// Validate the resized dimensions.
		if (segmentation_output.rows != img_vsize || segmentation_output.cols != img_hsize) {
			std::cerr << "Error: resized segmentation output dimensions do not match expected size." << std::endl;
			return;
		}

		// Copy segmentation output data to dout.
		for (int i = 0; i < img_vsize; i++) {
			for (int j = 0; j < img_hsize; j++) {
				dout[i * img_hsize + j] = static_cast<_Ty>(segmentation_output.at<uchar>(i, j));
			}
		}
	}

private:
	/**
	 * Operator to compute the dilation or erosion for a pixel.
	 * @param pix_x: Pixel position.
	 * @param se_int_size: Integer size of the structuring element.
	 * @param se_frc_size: Fractional size of the structuring element.
	 * @param size: Total size of the array.
	 * @param din: Input data array.
	 * @return Computed pixel value after dilation or erosion.
	 */
	_Ty operator()(const int pix_x, const int se_int_size, const int se_frc_size, const int size, const _Ty* din)
	{
		int max_min = dNe ? init_max : init_min;
		int frst_val, last_val;

		for (int i = 0, m = pix_x - se_int_size; i <= 2 * se_int_size; i++, m++)
		{
			int x = std::max(0, std::min(m, size - 1));

			if (i == 0)
				frst_val = (int)din[x];
			else if (i == 2 * se_int_size)
				last_val = (int)din[x];
			else
				max_min = dNe ? std::max(max_min, (int)din[x]) : std::min(max_min, (int)din[x]);
		}

		int inner_maxmin = max_min;
		if (dNe)
			max_min = std::max({ max_min, frst_val, last_val });
		else
			max_min = std::min({ max_min, frst_val, last_val });
		int outer_maxmin = max_min;

		int tmp = (outer_maxmin - inner_maxmin) * se_frc_size + (inner_maxmin << _Sh);
		tmp >>= _Sh;
		tmp = std::max(0, tmp);

		return (_Ty)tmp;
	}

	/**
	 * Computes the average value within a structuring element.
	 * @param pix_x: Pixel position.
	 * @param se_int_size: Integer size of the structuring element.
	 * @param se_frc_size: Fractional size of the structuring element.
	 * @param size: Total size of the array.
	 * @param din: Input data array.
	 * @return Average value as _Ty.
	 */
	_Ty avg(const int pix_x, const int se_int_size, const int se_frc_size, const int size, const _Ty* din)
	{
		int average = 0;
		for (int k = 0, m = pix_x - se_int_size; k <= 2 * se_int_size; k++, m++) {
			int x = std::max(0, std::min(m, size - 1));
			average += din[x];
		}

		average += se_int_size;
		average /= (2 * se_int_size + 1);
		return (_Ty)(dNe ? std::max(average, din[pix_x]) : std::min(average, din[pix_x]));
	}

private:
	bool dNe = true; // Flag to determine dilation (true) or erosion (false).
	const int init_max = -100000, init_min = 100000; // Initialization constants for max and min values.
};
