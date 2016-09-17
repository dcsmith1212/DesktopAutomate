#ifndef TEXTBOXFINDER_H 
#define TEXTBOXFINDER_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <baseapi.h>
#include <allheaders.h>

using std::vector;
using cv::Mat;
using cv::Point;

// decomposeimage.cpp
void preprocessing(Mat&, Mat&);
void decompose_image(Mat&, Mat&, Mat&);

// enhanceimage.cpp
void enhance_image(Mat&, Mat&);

// findquery.cpp
cv::Mat find_query_image(Mat& fullScreen, Mat& queryImg, Mat& drawImg);

// findrectangle.cpp
double angle(Point, Point, Point);
void findSquares(const Mat&, vector<vector<Point>>&);
void drawSquares(Mat&, const vector<vector<Point>>&);

// linuxinterface.cpp
void get_input_images(Mat&, Mat&); 

// linuxscreen.cpp
void ImageFromDisplay(vector<uint8_t>&, int&, int&, int&);

// tessocr.cpp
char* callTesseract(Mat); 

#endif
