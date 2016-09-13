// g++ -std=c++11 boxFinder.cpp -I/usr/include/leptonica -I/usr/include/tesseract/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract
#include <iostream>
#include <string>
#include "findRectangles.cpp"
#include "opencv2/opencv.hpp"
#include <baseapi.h>
#include <allheaders.h>

// Standard functions/data types
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

// OpenCV functions/datatypes
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::cvtColor;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::Scalar;
using cv::imread;
using cv::Size;
using cv::Range;

char* callTesseract(Mat);

int main() {
	// Screenshot
	string inputImg = "images/screen4.png";
	Mat fullScreen = imread(inputImg, CV_LOAD_IMAGE_COLOR);

	// Query image
	string inputSub = "images/textbox4.png";
	Mat subImage = imread(inputSub, CV_LOAD_IMAGE_COLOR);

	namedWindow("Display", 0);
	imshow("Display", fullScreen);

	// Runs NCC on query image in screenshot
	Mat output;
	cv::matchTemplate(fullScreen, subImage, output, CV_TM_CCOEFF_NORMED);
	
	// Finds best match point
	double min, max;
	Point minLoc, maxLoc;
	cv::minMaxLoc(output, &min, &max, &minLoc, &maxLoc);

	// For rectangle definition
	Point bottomRight;
	bottomRight.x = maxLoc.x + subImage.cols;
	bottomRight.y = maxLoc.y + subImage.rows;
	Rect subset(maxLoc, bottomRight);

	// Samples the screenshot with found query image
	Mat subsetImg(fullScreen, subset);

	namedWindow("SubImg", 0);
	imshow("SubImg", subsetImg);

	// Converts query image to grayscale
	Mat gray;
	Mat squareImg = subsetImg.clone();
	Mat fieldParent = subsetImg.clone();
	cvtColor(subsetImg, gray, CV_BGR2GRAY);



	// Converts lap and gray to same type, then find Laplacian of query image
	Mat lap;
	lap.convertTo(lap, CV_8U);
	gray.convertTo(gray, CV_8U);

	Mat labeled = gray.clone();
	
	cv::Laplacian(gray,lap,0);	
	namedWindow("Gray", 0);
	imshow("Gray", gray);


	//namedWindow("Stretched", 0);
	//imshow("Stretched", stretched);

	//namedWindow("Lap", 0);
	//imshow("Lap", lap);	

	// Binarizes Laplacian to emphasize text field boundary
	Mat binary;
	cv::threshold(lap, binary, 10, 255, CV_THRESH_BINARY);

	//namedWindow("Bin", 0);
	//imshow("Bin", binary);

	// Find rectangle that makes up boundary of text field
	vector<vector<Point>> rectangles;
	findSquares(binary, rectangles);

	// Number of potential text fields
	cout << "# of potential text fields: " << rectangles.size() << endl;
	//if (rectangles.size() > 1) cerr << "Too many rectangles to choose from!" << endl;
	drawSquares(squareImg, rectangles);


	// Sample the query image further, leaving only the text field
	Rect textFieldBb = cv::boundingRect(rectangles[0]);
	Mat textField(fieldParent, textFieldBb);



	// All the displayed images (temporary)
	namedWindow("TF", 0);
	imshow("TF", textField);	


	// "White out" of textfield to leave only label
	int nRows = textFieldBb.height;
	int nCols = textFieldBb.width;
	Mat blank(nRows, nCols, CV_8U, Scalar(255));

	namedWindow("Blank", 0);
	imshow("Blank", blank);

	//Range rowRange(textFieldBb.x, textFieldBb.x + textFieldBb.width - 1);
	//Range colRange(textFieldBb.y, textFieldBb.y + textFieldBb.height - 1);

	Mat roi(labeled(textFieldBb));
	blank.copyTo(roi);

        namedWindow("Lab", 0);
        imshow("Lab", labeled);

	Mat scaledLab, enhancedLab;
        resize(labeled, scaledLab, Size(), 5, 5);
        cv::GaussianBlur(scaledLab, enhancedLab , cv::Size(0, 0), 3);
        cv::addWeighted(scaledLab, 1.5, enhancedLab, -0.5, 0, enhancedLab);

	char* outLabel = callTesseract(enhancedLab);
	cout << "TextBox label:   " << outLabel << endl;



	// Enhancing text field and sending to tesseract
	Mat scaled, enhanced;
	resize(textField, scaled, Size(), 5, 5);

	
	//namedWindow("SCALED", 0);
	//imshow("SCALED", scaled);
	
	cv::GaussianBlur(scaled, enhanced , cv::Size(0, 0), 3);
	cv::addWeighted(scaled, 1.5, enhanced, -0.5, 0, enhanced);

	namedWindow("Enhanced", 0);
	imshow("Enhanced", enhanced);
	
	char* outText = callTesseract(enhanced);

	cout << "TextBox data:   " << outText << endl;
	

	waitKey(0);
	return 0;
}


char* callTesseract(Mat inputImg) {
	char *outText;

	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (api->Init(NULL, "eng")) {
        	fprintf(stderr, "Could not initialize tesseract.\n");
        	exit(1);
	}


	// Open input image with leptonica library
	api->SetImage((uchar*)inputImg.data, inputImg.size().width, inputImg.size().height, inputImg.channels(), inputImg.step1());
	// Get OCR result
	outText = api->GetUTF8Text();
	//printf("OCR output:\n%s", outText);

	return outText;
	// Destroy used object and release memory
	//api->End();
	//delete [] outText;
	
}
