#include "textboxfinder.h"
#include <string>
#include <iostream>

using std::cout;
using std::cin;
using std::endl;
using std::string;
using cv::imread;

void get_input_images(Mat& fullScreen, Mat& queryImage) {
	cout << "\n-----------------------------------\n"
		"Textbox Location and Data Retrieval\n"
		"   Implementation as of 9/16/16    \n"
		"-----------------------------------\n\n"
		"Which image pair would you like to test?\n"
		" (enter name of one of these directories):\n\n";
	system("ls images/textboxes/");
	cout << ">> ";

	string testName;
	cin >> testName;

	cout << "images/textboxes/" + testName + "/screen.png" << endl;

	fullScreen = imread("images/textboxes/" + testName + "/screen.png", CV_LOAD_IMAGE_COLOR);
	queryImage = imread("images/textboxes/" + testName + "/template.png", CV_LOAD_IMAGE_COLOR);	
}
