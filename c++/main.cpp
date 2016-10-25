#include <iostream>
#include <string>
#include "findboxbytemplate.h"
#include "textdetect.h"
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

void findComponentCenters(Mat& input_img, vector<Point>& box_centers, vector<Rect>& bounding_boxes, vector<vector<Point> >& pos_contours, vector<vector<Point> >& neg_contours) {
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
      //showim("neg", neg);

      cv::imwrite("pos.png", pos);
      cv::imwrite("neg.png", neg);
      
      // Find the resulting connected components in the general gradient
      pos.convertTo(pos, CV_8UC1);
      neg.convertTo(neg, CV_8UC1);
      cv::findContours(pos, pos_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
      cv::findContours(neg, neg_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

      // Store positive and negative bboxes in a vector 
      for (vector<Point> contour : pos_contours)
            bounding_boxes.push_back(cv::boundingRect(contour));
      for (vector<Point> contour : neg_contours)
            bounding_boxes.push_back(cv::boundingRect(contour));

      // Find the centers of the bounding boxes 
      for (Rect box : bounding_boxes) {
            Point center;
            center.x = (int)((box.x + box.width)/2);
            center.y = (int)((box.y + box.height)/2);
            box_centers.push_back(center);      
      }
}


void groupConnectedComponents(vector<std::unordered_set<int>>& candidate_groups,
                              std::unordered_set<int>& voided_groups,
                              vector<Rect>& bounding_boxes,
                              vector<Point>& box_centers,
                              vector<vector<Point> >& pos_contours,
                              vector<vector<Point> >& neg_contours,
                              vector<Rect>& candidate_boxes) {
     const int MAX_HORIZ = 7;
     const int MAX_VERT  = 4;
     int xdist, ydist;
     int g_id;
     int g_id2;
     std::unordered_set<int> group;
     std::unordered_set<int> group2;
     bool in_a_set;
     std::unordered_set<int> all_grouped;
     std::unordered_map<int,int> cc_group_ids;


     for (int i = 0; i < bounding_boxes.size(); i++) {
            for (int j = i+1; j < bounding_boxes.size(); j++) {
                  if ( (all_grouped.find(i) != all_grouped.end()) && \
                       (all_grouped.find(j) != all_grouped.end()) && \
                       (cc_group_ids.at(i) == cc_group_ids.at(j)) ) continue;
                  else {
                        xdist = std::abs(box_centers[i].x - box_centers[j].x);
                        ydist = std::abs(box_centers[i].y - box_centers[j].y);
                        if (( xdist < MAX_HORIZ ) && ( ydist < MAX_VERT )) {
                              if ( (all_grouped.find(i) != all_grouped.end()) && \
                                   (all_grouped.find(j) != all_grouped.end()) && \
                                   (cc_group_ids.at(i) != cc_group_ids.at(j)) ) {
                                    g_id = cc_group_ids.at(i);
                                    group = candidate_groups[g_id];

                                    g_id2 = cc_group_ids.at(j);
                                    group2 = candidate_groups[g_id2];
                                    for (int ccid : group2) cc_group_ids[ccid] = g_id;
                                    group.insert(group2.begin(), group2.end());

                                    candidate_groups[g_id] = group;
                                    voided_groups.insert(g_id2);
                              } else if (all_grouped.find(i) != all_grouped.end()) {
                                    g_id = cc_group_ids.at(i);
                                    group = candidate_groups[g_id];
                                    group.insert(j);
                                    candidate_groups[g_id] = group;

                                    all_grouped.insert(j);
                                    cc_group_ids.insert( std::pair<int,int>(j,g_id) );
                              } else if (all_grouped.find(j) != all_grouped.end()) {
                                    g_id = cc_group_ids.at(j);
                                    group = candidate_groups[g_id];
                                    group.insert(i);
                                    candidate_groups[g_id] = group;

                                    all_grouped.insert(i);
                                    cc_group_ids.insert( std::pair<int,int>(i,g_id) );
                              } else {
                                    std::unordered_set<int> new_group;
                                    new_group.insert(i);
                                    new_group.insert(j);
                                    candidate_groups.push_back(new_group);
                                    
                                    all_grouped.insert(i);
                                    all_grouped.insert(j);
                                    cc_group_ids.insert( std::pair<int,int>(i,candidate_groups.size()-1) );
                                    cc_group_ids.insert( std::pair<int,int>(j,candidate_groups.size()-1) );
                              }                        
                        }
                  }
            }
      }

      srand(time(NULL));
      
      const float MAX_AR = 10;
      const float MIN_AR = 0.5;
      
      // Find this bounding box for each of the groupings
      pos_contours.insert(pos_contours.end(), neg_contours.begin(), neg_contours.end());
      for (int i = 0; i < candidate_groups.size(); i++) {
            if (voided_groups.find(i) == voided_groups.end()) { 
                  if (candidate_groups[i].size() > 4) {
                        group = candidate_groups[i];
                        vector<Point> contour_group;
                        for (int id : group) {
                              vector<Point> contour = pos_contours[id];
                              contour_group.insert(contour_group.end(), contour.begin(), contour.end());
                        }
                        Rect big_box = cv::boundingRect(contour_group);
                        float ar = (float)big_box.width / (float)big_box.height;
                        float area = (float)big_box.width * (float)big_box.height;
                        if (ar > MIN_AR)  candidate_boxes.push_back(big_box);            
                  }
            }
      }
}



// Trying the AA-based technique
int main(int argc, char *argv[]) {
tic();
      Mat input_img = imread("google2.png");
      Mat pos_img = input_img.clone();
      Mat neg_img = input_img.clone();
      Mat group_img = input_img.clone();
      Mat valid_img = input_img.clone();
      Mat sample_img = input_img.clone();
	Mat match_img = input_img.clone();
      //showim("input", input_img);

      vector<vector<Point> > pos_contours, neg_contours;
      vector<Point> box_centers;
      vector<Rect> bounding_boxes;
      findComponentCenters(input_img, box_centers, bounding_boxes, pos_contours, neg_contours);

      // Group the connect components based on proximity
      vector<std::unordered_set<int>> candidate_groups;
      std::unordered_set<int> voided_groups;
      vector<Rect> candidate_boxes;
      groupConnectedComponents(candidate_groups, voided_groups, bounding_boxes, 
                               box_centers, pos_contours, neg_contours, candidate_boxes);
      
      // Display bounding boxes of groups
      for (Rect big_box : candidate_boxes)
            cv::rectangle(group_img, big_box, CV_RGB(rand()%256, rand()%256, rand()%256), 2);
      cv::imwrite("groups.png", group_img);
      showim("Candidate groups", group_img);

toc();

tic();

      char* query = argv[1];

      int best_ind = 0;
      char* best_match;
      int best_score = 99;

      int curr_score;
      
      vector<Rect> valid_boxes;
      for (int i = 0; i < candidate_boxes.size(); i++) {
            Mat subsample(sample_img, candidate_boxes[i]);
			cv::cvtColor(subsample, subsample, CV_RGB2GRAY);
            //vector<Mat> channels;
            //cv::split(subsample, channels);
            //showim("red", channels[0]);
            
            // Calculate histogram for this potential text region
            int hist_size = 256;
            float range[] = {0, 255};
            const float *ranges[] = {range};
            MatND hist;
            cv::calcHist(&subsample, 1, 0, Mat(), hist, 1, &hist_size, ranges, true, false);
      
            // Place a 1 at histogram peaks (>6)
            // Should be ~10 for AA text
            vector<bool> parts;
            float binVal;
            for (int h = 0; h < hist_size; h++) {
                  binVal = hist.at<float>(h);
                  if (binVal >= 6) parts.push_back(1);
                  else parts.push_back(0);
            }
/*
            // Count number of groupings of sequential nonzero bins
            // (number of peaks)
            int n_peaks = 0;
            int n_sequential = 0;
            bool on_a_peak = 0;
            bool contains_aa = 1;

            for (bool val : parts) {
                  if (!on_a_peak) {
                        if (val) {
                              n_peaks++;
                              n_sequential++;
                              on_a_peak = 1;
                        }
                  } else {
                        if (val) {
                              n_sequential++;
                              if (n_sequential > 6) {
                                    contains_aa = 0;
                                    break;
                              }
                        } else { 
                              on_a_peak = 0;
                              n_sequential = 0;
                        }
                  }
            }
*/            
            valid_boxes.push_back(candidate_boxes[i]);
      
            
            float curr_val;
            float left_max = 0;
            int left_ind = 0;
            float right_max = 0;
            int right_ind = hist_size - 1;
            for (int k = 0; k < hist_size; k++) { 
                  curr_val = hist.at<float>(k);
                  if (curr_val >= left_max) {
                        left_max = curr_val;
                        left_ind = k;
                  } else break;
            }
            for (int k = hist_size - 1; k >= 0; k--) {
                  curr_val = hist.at<float>(k);
                  if (curr_val >= right_max) {
                        right_max = curr_val;
                        right_ind = k;
                  } else break;
            }

			int bg_color, fg_color;
			if (std::max(left_max, right_max) == right_max) {
				bg_color = right_ind;
				fg_color = left_ind;
			} else {
				bg_color = left_ind;
				fg_color = right_ind;
			}

            int color_spread = std::abs(bg_color - fg_color);
      
            Mat fg_mask, bg_mask, inter_mask;
            if (bg_color > fg_color) { // Dark text on light
                  bg_mask = subsample > bg_color - 0.25*color_spread;
                  fg_mask = subsample < fg_color + 0.40*color_spread;
            } else {
                  bg_mask = subsample < bg_color + 0.25*color_spread;
                  fg_mask = subsample > fg_color - 0.25*color_spread;
            }
            

            //cv::bitwise_or(bg_mask, fg_mask, inter_mask);
            //cv::bitwise_not(inter_mask, inter_mask);
     
            //Mat c = 255.0 / (float)(bg_color - fg_color) * (subsample - fg_color);
            //Mat c2;
            //inter_mask.convertTo(inter_mask, CV_8UC1);
            //c.copyTo(c2, inter_mask);

            //Mat decolored = 255.0*bg_mask + 0.0*fg_mask + inter_mask;
            //showim("decolored", decolored);
     
            Mat upscaled_text;
            int scale_factor = 3;
            cv::bitwise_not(fg_mask, fg_mask);
            cv::resize(fg_mask, upscaled_text, cv::Size(0,0), scale_factor, scale_factor, INTER_NEAREST);
            //showim("upscaled", upscaled_text);
            char* text = callTesseract(upscaled_text);

            curr_score = levenshteinDistance(query, text);
            if (curr_score < best_score) {
                  best_score = curr_score;
                  best_ind = i;
                  best_match = text;
            }
      
      }
      
      cout << "Query:        " << query << endl;
      cout << "Best match:   " << best_match << endl;

      cv::rectangle(match_img, candidate_boxes[best_ind], CV_RGB(0,255,0), 2);
      showim("groups", match_img);
   


      
/*
      int hist_w = 512; int hist_h = 400;
      int bin_w = cvRound( (double)hist_w/hist_size );
      Mat hist_img(hist_h, hist_w, CV_8UC1, cv::Scalar(0,0,0));
      normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX, -1, Mat());
      for (int i = 1; i < hist_size; i++) {
            cv::line(hist_img, Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
                               Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
                               cv::Scalar(255,0,0), 1, 8, 0);
      }
      showim("histogram", hist_img);
*/

/*
      // Draw positive and negative contours
      cv::drawContours(pos_img, pos_contours, -1, CV_RGB(255,0,0));
      cv::drawContours(neg_img, neg_contours, -1, CV_RGB(255,0,0));
      showim("pos_conts", pos_img);
      showim("neg_conts", neg_img); 
*/

      //for (Rect box : valid_boxes)
      //      cv::rectangle(valid_img, box, CV_RGB(255,0,0),1);
      //showim("Valid groups", valid_img);



toc();


      waitKey(0);
      return 0;
}
