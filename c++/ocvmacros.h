#ifndef OCVMACROS_H
#define OCVMCROS_H

#include "opencv2/opencv.hpp"

#include <iostream>
#include <stack>
#include <ctime>
#include <string>

void showim(std::string windowName, cv::Mat image); 
void tic();
void toc();


#endif
