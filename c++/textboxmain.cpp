#include <iostream>
#include <string>
#include "ocvmacros.h"
#include "textboxfinder.h"

// Standard functions/datatypes
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
	// Gets user's desired fullscreen and query image
	Mat fullScreen, queryImage;
	get_input_images(fullScreen, queryImage);

	// Display chosen images
	showim("Fullscreen", fullScreen);
	showim("Query Image", queryImage);

	// Runs NCC on query image in screenshot
	Mat drawImg = fullScreen.clone();
	Mat foundImage = find_query_image(fullScreen, queryImage, drawImg);	
	showim("Found Image", foundImage);
	showim("Edited fullscreen", drawImg);

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

	cout << endl << "Tesseract OCR Output:" << endl; 
	char* outLabel = callTesseract(enhancedLabelImg);
	cout << "TextBox label:   " << outLabel;
	
	char* outText = callTesseract(enhancedFieldImg);
	cout << "TextBox data:   " << outText;


	cout << endl << endl << "Press any key to quit." << endl;
	waitKey(0);
	return 0;
}




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



// g++ -std=c++14 boxFinder.cpp -I/usr/include/leptonica -I/usr/include/tesseract/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract

// Current dependencies:
//
// OpenCV
// Leptonica
// Tesseract
// ffmpeg

// TO DO:
//  - Get Makefile working
//  - Return coordinates

