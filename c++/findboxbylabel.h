#ifndef FIND_BY_LABEL_H
#define FIND_BY_LABEL_H

#include <iostream>
#include <string>
#include "findboxbytemplate.h"
#include "ocvmacros.h"
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>

#include <cmath>
#include <stdlib.h>
#include <time.h>
// Standard functions/datatypes
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::cerr;

// OpenCV functions/datatypes
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::imread;
using cv::Point;
using cv::Range;
using cv::Rect;

double angle( Point pt1, Point pt2, Point pt0 );
void findRectangles(const Mat& image, vector<Rect>& rectangles, bool squaresOnly);
void drawRectangles(Mat& image, const vector<Rect>& rectangles);
      
class FindBoxByLabel {
private:
	string queried_label;
	Mat input_img;
	Mat group_img;				
	Mat valid_img;					
	Mat sample_img;				
	Mat match_img;
      Mat rectangle_img;
      Mat rect_display;
      vector<vector<Point> > pos_contours, neg_contours;	
	vector<Point> box_centers;							
	vector<Rect> bounding_boxes;							
	vector<std::unordered_set<int>> candidate_groups;		
	std::unordered_set<int> voided_groups;				
	vector<Rect> candidate_boxes;							
      vector<Rect> valid_boxes;
      Mat preprocessed_text_img;
      vector<string> matched_text;
      vector<Rect> rectangles;

      int object_id;

	int hist_size = 256;
      int bg_color, fg_color;

	int best_ind = 0;
	char* best_match;
	int best_score = 9999999;
	int curr_score;

	inline char* callTesseract(Mat& input);
      int lcSubStr(string X, string Y);
      int levenshteinDistance(string query_str, string test_str);
	void findComponentCenters();
	void groupConnectedComponents();
      bool displayHistogram(cv::MatND hist, string label);
      bool filterByHistogram(Mat subsample);
      void calculateHistogramExtrema(Mat candidate_img);
	void preprocessForOCR(Mat candidate_img);
      void locateBestFieldMatch(int best_ind);

public:
	FindBoxByLabel(int argc, char *argv[], string object);
	void findBoxByLabel();
};

#endif
