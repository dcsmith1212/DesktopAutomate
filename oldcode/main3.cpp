#include <iostream>
#include <string>
#include "findboxbytemplate.h"
#include "textdetect.h"
#include "ocvmacros.h"
#include <map>

#include <cmath>

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

using cv::Range;

using namespace cv;

inline char* callTesseract(Mat input_image) {
      tesseract::TessBaseAPI api;

      // Initialize tesseract-ocr with English, without specifying tessdata path
      if (api.Init(NULL, "eng")) {
	    fprintf(stderr, "Could not initialize tesseract.\n");
	    exit(1);
      }

      // Open input image with leptonica library
      api.SetImage((uchar*)input_image.data, input_image.size().width, \
                  input_image.size().height, input_image.channels(), input_image.step1());

      // Return OCR result
      return api.GetUTF8Text();
}


int levenshteinDistance(const char* query, const char* test) {
      string query_str(query);
      string test_str(test);
      int m = query_str.length();
      int n = test_str.length();

      int dist_arr[m+1][n+1] = {0};      
      
      for (int i = 1; i <= m; i++) dist_arr[i][0] = i;
      for (int j = 1; j <= n; j++) dist_arr[0][j] = j;

      int subst_cost;
      for (int j = 1; j <= n; j++) {
            for (int i = 1; i <= m; i++) {
                  if (query_str[i-1] == test_str[j-1]) subst_cost = 0;
                  else subst_cost = 1;
                  dist_arr[i][j] = std::min( std::min(dist_arr[i-1][j] + 1,  \
                                                      dist_arr[i][j-1] + 1), \
                                             dist_arr[i-1][j-1] + subst_cost );
            }
      }
      
      return dist_arr[m][n];
}

// TODO: Take a subimage, binarize (inversely), then find the connected components. Use a histogram to find those components that correspond to letters (based on word dimensions and area, etc.). Then count those components. If the number falls within, say, +- 2 of the correct number of letters in the query word, pass it to Tesseract. Or maybe keep up the identification with scalar properties.

// Could make a template function
// Does this function exist already?
// TODO: If multiple indices have the same score, store all of them. Then do further processing to determine the correct word
int minScore(vector<int> scores) {
      int min = scores[0];
      int mindex = 0;
      for (int i = 1; i < scores.size(); i++) {
            if (scores[i] < min) {
                  min = scores[i];
                  mindex = i;
            }
      }
}

int main(int argc, char *argv[]) {
// This implementation calls Tesseract on each of the subimages and finds
// the Levenshtein distance between the result and the original query word.

	vector<Rect> validBoundingBoxes;
	Rect currBox;
	float aspect_ratio;	
	Range rowRange;
	Range colRange;
      Mat up_scale;

      char* query = "FEMM Preferences";

      std::vector<int> word_ids;
      std::vector<int> word_scores;
      std::vector<char*> words;

      char* tessOutput;
	for (int i = 0; i < boundingBoxes.size(); i++) {
		currBox = boundingBoxes[i];
		aspect_ratio = (float)currBox.width / (float)currBox.height;
		if (aspect_ratio > 0.2) {
			colRange.start = std::max(boundingBoxes[i].x - 5, 0);
			rowRange.start = std::max(boundingBoxes[i].y - 5, 0);
			rowRange.end = std::min(rowRange.start + boundingBoxes[i].height \
                                                         + 10, sampleImg.rows);
			colRange.end = std::min(colRange.start + boundingBoxes[i].width  \
                                                         + 10, sampleImg.cols);
			
			//rectangle(valid, sampleRect, CV_RGB(0,0,255));
			Mat roi(sampleImg,rowRange,colRange);
                  cv::cvtColor(roi, roi, CV_BGR2GRAY);
                  cv::threshold(roi, roi, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
                       
                  resize(roi, up_scale, cv::Size(0,0), 4.0, 4.0);      

                  tessOutput = callTesseract(roi);
//			cout << "Line " << i << ":  " << tessOutput << endl;

                  int score = levenshteinDistance(query, tessOutput);
                  word_ids.push_back(i);
                  word_scores.push_back(score);
                  words.push_back(tessOutput); 


                  // When comparing user input to list of strings, first get rid of all upper case letters, then remove the \n characters from all the strings
                  // Also, prior to matching, filter out those blocks that have an AR that's inconsistent with the word
                  // Also, filter out blank strings first
                  // MINIMIZE # OF CALLS TO TESSERACT
		}
	}

      int mindex = minScore(word_scores);
      cout << words[mindex] << endl;

      rectangle(valid, boundingBoxes[mindex], CV_RGB(0,255,0), 2);

	imwrite("valid.png", valid);


      waitKey(0);
 	return 0;

}
