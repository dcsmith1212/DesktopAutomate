#ifndef FINDRECTANGLES_H
#define FINDRECTANGLES_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <string.h>

using std::vector;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( cv::Point, cv::Point, cv::Point);

void findSquares( const cv::Mat&, vector<vector<cv::Point> >& );

void drawSquares( cv::Mat&, const vector<vector<cv::Point> >& );

#endif
