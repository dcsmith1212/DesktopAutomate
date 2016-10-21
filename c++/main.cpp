#include <iostream>
#include <string>
#include "findboxbytemplate.h"
#include "textdetect.h"
#include "ocvmacros.h"
#include <map>
#include <set>

#include <cmath>

// Standard functions/datatypes
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::cerr;
using std::set;

// OpenCV functions/datatypes
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::imread;
using cv::Point;
using cv::Range;
using cv::Rect;

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

// Trying the AA-based technique
int main(int argc, char *argv[]) {
      Mat input_img = imread("google.png");
      Mat pos_img = input_img.clone();
      Mat neg_img = input_img.clone();
      Mat group_img = input_img.clone();
      //showim("input", input_img);

      // To grayscale
      cv::cvtColor(input_img, input_img, CV_RGB2GRAY);
      
      // Calculate positive, negative and total horizontal gradient
      const int KERNEL_SIZE = 1;
      Mat pos;
      cv::Sobel(input_img, pos, CV_64F, 1, 0, KERNEL_SIZE);
      Mat both = abs(pos);
      Mat neg = both - pos;

      //showim("both", both);
      //showim("pos", pos);
      showim("neg", neg);

      
      // Find the resulting connected components in the general gradient
      pos.convertTo(pos, CV_8UC1);
      neg.convertTo(neg, CV_8UC1);
      vector<vector<Point> > pos_contours, neg_contours;
      cv::findContours(pos, pos_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
      cv::findContours(neg, neg_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

      cout << "# of pos:  " << pos_contours.size() << endl;
      cout << "# of neg:  " << neg_contours.size() << endl << endl;
      
      // Store positive and negative bboxes in a vector 
      vector<Rect> bounding_boxes;
      for (vector<Point> contour : pos_contours)
            bounding_boxes.push_back(cv::boundingRect(contour));
      for (vector<Point> contour : neg_contours)
            bounding_boxes.push_back(cv::boundingRect(contour));
     
      // Find the centers of the bounding boxes 
      vector<Point> box_centers;
      for (Rect box : bounding_boxes) {
            Point center;
            center.x = (int)((box.x + box.width)/2);
            center.y = (int)((box.y + box.height)/2);
            box_centers.push_back(center);      
      }

      // Group the connect components based on proximity
      const int MAX_HORIZ = 8;
      const int MAX_VERT  = 7;
      vector<set<int>> valid_groups;
      set<int> empty;
      valid_groups.push_back(empty);
      bool in_a_set;
      set<int> group;
      set<int> all_grouped;
      
      int xdist, ydist;
      for (int i = 0; i < bounding_boxes.size(); i++) {
            for (int j = 0; j < bounding_boxes.size(); j++) {
                  if (i != j) {
//                  printf("Current sets:\n");
//                  for (set<int> group : valid_groups) {
//                        cout << "{";
//                        for (int cc : group) cout << cc << ", ";
//                        cout << "}" << endl;
//                  }
//                  printf("Pair: (%d,%d)\n", i, j);
                        xdist = std::abs(box_centers[i].x - box_centers[j].x);
                        ydist = std::abs(box_centers[i].y - box_centers[j].y);
//                  printf("Dists: x = %d,  y = %d\n", xdist, ydist);
                        if (( xdist < MAX_HORIZ ) && ( ydist < MAX_VERT )) {
//                  printf("Within distance\n");
                              in_a_set = 0;
                              for (int k = 0; k < valid_groups.size(); k++) {
                                    group = valid_groups[k];
                                    if ( group.find(i) != group.end() && \
                                         group.find(j) != group.end() ) {
//                  printf("Both components in a set \n");
                                          in_a_set = 1;
                                          break;
                                    } else if ( group.find(i) != group.end() ) {
//                  printf("i = %d  in a set\n", i);
//                  cout << "Group before addition:   S = {";
//                  for (int cc : group) cout << cc << ", ";
//                  cout << "}" << endl;
                                          group.insert(j);
                                          valid_groups[k] = group;
//                  cout << "Group after addition:   S = {";
//                  for (int cc : group) cout << cc << ", ";
//                  cout << "}" << endl;
                                          in_a_set = 1;
                                          break;
                                    } else if ( group.find(j) != group.end() ) {
//                  printf("j = %d  in a set\n", j);
                                          group.insert(i);
                                          valid_groups[k] = group;
                                          in_a_set = 1;
                                          break;
                                    }
                              }
                              if (!in_a_set) {
//                  printf("Neither of components in a set\n");
                                    set<int> new_group;
                                    new_group.insert(i);
                                    new_group.insert(j);
                                    valid_groups.push_back(new_group);
//                  cout << "New set added:   S = { ";
//                  for (int cc : new_group) cout << cc << ", ";
//                  cout << "}" << endl;
                              }                        
                        }
//                  printf("\n"); 
//                  waitKey(0);
                  }
            }
      }

      pos_contours.insert(pos_contours.end(), neg_contours.begin(), neg_contours.end());
      for (set<int> group : valid_groups) {
            if (!group.empty()) {
                  vector<Point> contour_group;
                  for (int id : group) {
                        vector<Point> contour = pos_contours[id];
                        contour_group.insert(contour_group.end(), contour.begin(), contour.end());
                  }
                  Rect big_box = cv::boundingRect(contour_group);
                  cv::rectangle(group_img, big_box, CV_RGB(255,0,0), 2);
            }
      }    
      showim("groups", group_img); 
/*
      // Draw positive and negative contours
      cv::drawContours(pos_img, pos_contours, -1, CV_RGB(255,0,0));
      cv::drawContours(neg_img, neg_contours, -1, CV_RGB(255,0,0));
      showim("pos_conts", pos_img);
      showim("neg_conts", neg_img); 
*/




      waitKey(0);
      return 0;
}
