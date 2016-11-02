#include <iostream>
#include <string>
#include "findboxbylabel.h"
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

// AA-based technique
int main(int argc, char *argv[]) {
      FindBoxByLabel finder(argc, argv);
	finder.findBoxByLabel();

	waitKey(0);
	return 0;
}
