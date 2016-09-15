#include "findquery.h"

cv::Mat find_query_image(cv::Mat& fullScreen, cv::Mat& queryImg) {
	// Need to add functionality to return actual coordinates of query image
	cv::Mat output;
	cv::matchTemplate(fullScreen, queryImg, output, CV_TM_CCOEFF_NORMED);

	// Finds best match point
	double min, max;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(output, &min, &max, &minLoc, &maxLoc);

	// For rectangle definition
	cv::Point bottomRight;
	bottomRight.x = maxLoc.x + queryImg.cols;
	bottomRight.y = maxLoc.y + queryImg.rows;
	cv::Rect subset(maxLoc, bottomRight);

	// Samples the screenshot with found query image
	cv::Mat foundImg(fullScreen, subset);
	return foundImg;
}
