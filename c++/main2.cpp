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
// Find box by template
/*
      // Initialize finder object (empty constuctor will call getInputImages() automatically)
	FindBoxByTemplate finder;

	// Retrieves location of template in screenshot, and determines label and field text
	finder.findBoxByTemplate();
      cout << "Template location (UL corner):   (" << finder.template_stats.x << "," << finder.template_stats.y << ")" << endl;
      cout << "Template size:   (" << finder.template_stats.width << "x" << finder.template_stats.height << ")" << endl << endl;

      cout << "Tesseract output:" << endl;
      cout << "Textbox label:   " << finder.text_label << endl;
      cout << "Textbox field:   " << finder.text_field << endl;
*/
/*
      // Detect text in the image
      DetectText dt(argv);      
      dt.detectText();

	waitKey(0);
	return 0;
*/





      ///

      namedWindow("test", 0);
    Mat rawFrame = imread("images/textboxes/test09/screen.png");

    Mat contourImg = rawFrame.clone();
	Mat bboxImg = rawFrame.clone();
	Mat valid = rawFrame.clone();
	Mat sampleImg = rawFrame.clone();

	cv::Mat grayFrame;
	cv::Mat sobel1;
	cv::Mat sobel2;

	// Structuring elements for closing, opening, and tophat
	cv::Mat strel1(4,  4, CV_8U, cv::Scalar(1));
	cv::Mat strel2(12, 16, CV_8U, cv::Scalar(1));
	cv::Mat strel3(8, 8, CV_8U, cv::Scalar(1));

	// Scales the image down for processing
	cv::Mat processedFrame(rawFrame.rows, rawFrame.cols, CV_32FC1);
	rawFrame.convertTo(rawFrame, CV_32FC1, 1.0/255.0);		

    cv::cvtColor(rawFrame, grayFrame, CV_BGR2GRAY);
	cv::Sobel(grayFrame, sobel1, -1, 1, 0, 3, 1.0/2.0);	
	cv::Sobel(grayFrame, sobel2, -1, 0, 1, 3, 1.0/2.0);	
	cv::magnitude(sobel1, sobel2, processedFrame);
	cv::morphologyEx(processedFrame, processedFrame, cv::MORPH_CLOSE, strel1);
	Mat closed = processedFrame.clone();
	closed = closed * 255.0;
	imwrite("closed.png", closed);

	cv::morphologyEx(processedFrame, processedFrame, cv::MORPH_TOPHAT, strel2);
	cv::morphologyEx(processedFrame, processedFrame, cv::MORPH_OPEN, strel1);
	cv::GaussianBlur(processedFrame, processedFrame, cv::Size(0,0), 2, 2);
	processedFrame = 255.0 * processedFrame;
	cv::threshold(processedFrame, processedFrame, 70, 255, cv::THRESH_BINARY);
	cv::morphologyEx(processedFrame, processedFrame, cv::MORPH_OPEN, strel3);
	//cv::morphologyEx(processedFrame, processedFrame, cv::MORPH_DILATE, strel1);
	processedFrame.convertTo(processedFrame, CV_8UC1, 1.0);
      imwrite("test.png", processedFrame);


	vector<vector<Point> > contours;
	findContours(processedFrame, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	drawContours(contourImg, contours, -1, CV_RGB(255,0,0));

	vector<Rect> boundingBoxes;
	for (int i = 0; i < contours.size(); i++) {
		boundingBoxes.push_back(boundingRect(contours[i]));	
		
		rectangle(contourImg, boundingBoxes[i], CV_RGB(0,0,255));
	}
	imwrite("contour.png", contourImg);


      Rect currBox;
      float aspect_ratio;
      Range rowRange;
      Range colRange;
      Mat up_scale;
      namedWindow("subimage",0);

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
            
                  cv::imshow("subimage", up_scale);
                  waitKey(0);
            }
            
      }


      waitKey(0);
 	return 0;

}



// Captures screen shot
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


// Current dependencies:
//
// OpenCV
// Leptonica
// Tesseract
// Boost
