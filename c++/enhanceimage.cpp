#include "textboxfinder.h"

void enhance_image(Mat& inputImg, Mat& enhancedImg) {
	Mat scaledImg;
	cv::resize(inputImg, scaledImg, cv::Size(), 5, 5);
	cv::GaussianBlur(scaledImg, enhancedImg, cv::Size(0, 0), 3);
	cv::addWeighted(scaledImg, 1.5, enhancedImg, -0.5, 0, enhancedImg);
}
