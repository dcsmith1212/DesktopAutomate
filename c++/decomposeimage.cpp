#include "textboxfinder.h"
#include <vector>
#include <iostream>
using std::vector;
using cv::Point;
using cv::Rect;

const int BIN_THRESH = 10;

//WEIHVWIEVBIWEBV
Mat squareImg;
Mat fieldParent;
Mat labeled;

void preprocessing(Mat& foundImage, Mat& binary) {
	// Copy input image for sampling later in decompose_image
	squareImg = foundImage.clone();
	fieldParent = foundImage.clone();

	//Converts input image to grayscale
	Mat gray;
	cvtColor(foundImage, gray, CV_BGR2GRAY);

	// EIWVIWEBVIWEBV
	labeled = gray.clone();

	// Converts lap and gray to same type, then find Laplacian of query image
	Mat lap;
	lap.convertTo(lap, CV_8U);
	gray.convertTo(gray, CV_8U);
	cv::Laplacian(gray,lap,0);	

	// Binarizes Laplacian to emphasize text field boundary
	cv::threshold(lap, binary, BIN_THRESH, 255, CV_THRESH_BINARY);
}

void decompose_image(Mat& processedImg, Mat& textfieldImg, Mat& textboxLabel) {
	// Find rectangle that makes up boundary of text field
	vector<vector<Point>> rectangles;
	findSquares(processedImg, rectangles);

	// Number of potential text fields
	std::cout << "# of potential text fields: " << rectangles.size() << std::endl;
	drawSquares(squareImg, rectangles);

	// Sample the found image further, leaving only the text field
	// TODO: Instead of just taking first found rectangle, find the largest one
	Rect textFieldBb = cv::boundingRect(rectangles[0]);
	Mat textField(fieldParent, textFieldBb);

	textfieldImg = textField;

	// "White out" of textfield to leave only label
	int nRows = textFieldBb.height;
	int nCols = textFieldBb.width;
	Mat blank(nRows, nCols, CV_8U, cv::Scalar(255));
	Mat roi(labeled(textFieldBb));
	blank.copyTo(roi);

	textboxLabel = labeled;
}
