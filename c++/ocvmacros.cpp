#include "ocvmacros.h"

void showim(std::string windowName, cv::Mat image) {
	cv::namedWindow(windowName, 0);
	cv::imshow(windowName, image);
}
