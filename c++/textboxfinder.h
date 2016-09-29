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
using cv::Rect;

class TextBoxFinder
{
private:
      // Private member variables
      const int BINARY_THRESH = 10;
      const int CANNY_THRESH  = 50;
      const int N_THRESHOLDS  = 11;

      Mat template_img;
      Mat fullscreen_img;
      Mat found_img;
      Mat processed_img;
      Mat text_field_img;
      Mat text_label_img;

      Mat field_parent_img;
      Mat rect_display_img;
      Mat gray_clone_img;

      // findrectangle.cpp
      double angle(Point, Point, Point);
      void findRectangles(const Mat&, vector<vector<Point>>&, vector<double>&);
      void drawRectangles(Mat&, const vector<vector<Point>>&);

      // decomposeimage.cpp
      void preprocessImage();
      void decomposeImage();

      // enhanceimage.cpp
      void enhanceImage(Mat&);

      // findquery.cpp
      void findTemplateImage();

      // linuxscreen.cpp
      void imageFromDisplay(vector<uint8_t>&, int&, int&, int&);

      // tessocr.cpp
      char* callTesseract(Mat); 

public:
      // Constructors
      TextBoxFinder();
      TextBoxFinder(Mat&, Mat&);

      // Public variables
      char* text_field;
      char* text_label;
      Rect template_stats;

      // Finds label and text of a textbox given a template image
      void findBoxByTemplate(); 

      // linuxinterface.cpp
      void getInputImages(); 
};

#endif
