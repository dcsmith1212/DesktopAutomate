// g++ -std=c++14 boxFinder.cpp -I/usr/include/leptonica -I/usr/include/tesseract/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract

// Current dependencies:
//
// OpenCV
// Leptonica
// Tesseract
// ffmpeg
//
//

// TO DO:
//  - Group program into functions
//  - Get Makefile working
//  - Program UI
//  - Learn how findrectangles work (and clean up)

#include <iostream>
#include <string>
#include "ocvmacros.h"

#include "linuxscreen.h"
#include "tessocr.h"
#include "findquery.h"
#include "findtextbox.h"
#include "enhanceimage.h"


// Standard functions/data types
using std::cout;
using std::endl;
using std::string;
using std::vector;

// OpenCV functions/datatypes
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::imread;

int main() {
	// Screenshot
	string inputImg = "images/screen4.png";
	Mat fullScreen = imread(inputImg, CV_LOAD_IMAGE_COLOR);

	// Query image
	string inputSub = "images/textbox4.png";
	Mat subImage = imread(inputSub, CV_LOAD_IMAGE_COLOR);

	showim("Fullscreen", fullScreen);
	showim("Query Image", subImage);

	// Runs NCC on query image in screenshot
	Mat foundImage = find_query_image(fullScreen, subImage);	
	showim("Found Image", foundImage);

	// Processes the found image and then splits into textbox field 
	// and textbox label
	// Currently finds Laplacian and the binarizes with low threshold,
	// then uses rectangle-finding function to locate textbox field border 
	Mat binaryImg;
	preprocessing(foundImage, binaryImg);
	Mat textFieldImg, textLabelImg;
	decompose_image(binaryImg, textFieldImg, textLabelImg);

	Mat enhancedFieldImg, enhancedLabelImg;
	enhance_image(textFieldImg, enhancedFieldImg);
	enhance_image(textLabelImg, enhancedLabelImg);

	char* outLabel = callTesseract(enhancedLabelImg);
	cout << "TextBox label:   " << outLabel << endl;
	
	char* outText = callTesseract(enhancedFieldImg);
	cout << "TextBox data:   " << outText << endl;
	


/*
    int Width = 0;
    int Height = 0;
    int Bpp = 0;
    std::vector<std::uint8_t> Pixels;

    ImageFromDisplay(Pixels, Width, Height, Bpp);

    if (Width && Height)
    {
        Mat img = Mat(Height, Width, Bpp > 24 ? CV_8UC4 : CV_8UC3, &Pixels[0]); //Mat(Size(Height, Width), Bpp > 24 ? CV_8UC4 : CV_8UC3, &Pixels[0]); 

        namedWindow("WindowTitle", CV_WINDOW_AUTOSIZE);
        imshow("Display window", img);
    }
*/


	waitKey(0);
	return 0;
}
