#include "textboxfinder.h"
#include <iostream>
using std::cout;
using std::endl;

cv::Mat find_query_image(cv::Mat& fullScreen, cv::Mat& queryImg, cv::Mat& drawImg) {
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

	// Displays rectangle around found template area
	cv::rectangle(drawImg, maxLoc, bottomRight, CV_RGB(255,0,0));
	cout << "Found template image stats: " << endl;
	cout << "(X,Y) = (" << maxLoc.x << "," << maxLoc.y << ")" << endl;
	cout << "(W,H) = (" << subset.width << "," << subset.height << ")" << endl << endl;

	// Samples the screenshot with found query image
	cv::Mat foundImg(fullScreen, subset);
	return foundImg;
}
