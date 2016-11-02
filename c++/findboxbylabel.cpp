#include "findboxbylabel.h"
#include "persistence1d.hpp"

using cv::Scalar;

void preprocessImgForRectFinder(Mat& rectangle_img) {
      //Converts input image to grayscale
      Mat gray;
      cvtColor(rectangle_img, gray, CV_BGR2GRAY);

      // Converts lap and gray to same type, then find Laplacian of query image
      Mat lap;
      lap.convertTo(lap, CV_8UC1);
      gray.convertTo(gray, CV_8UC1);
      cv::Laplacian(gray,lap,0);	

      rectangle_img = lap;
}


// Helper function for findSquares:
// Finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 ) {
      double dx1 = pt1.x - pt0.x;
      double dy1 = pt1.y - pt0.y;
      double dx2 = pt2.x - pt0.x;
      double dy2 = pt2.y - pt0.y;
      return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// Returns sequence of squares detected on the image.
void findRectangles(const Mat& image, vector<Rect>& rectangles) {
      rectangles.clear();

      Mat bin;
      vector<vector<Point> > contours;

      // Find rectangles for a series of low threshold values
      for (int thresh = 5; thresh <= 75; thresh += 10) {
            bin = image >= thresh;
            //bin.convertTo(bin, CV_8UC1);
            
            // Find contours and store them all as a list
            cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // Test each contour
            for( int i = 0; i < contours.size(); i++ ) {
                  // Approximate contour with accuracy proportional
                  // to the contour perimeter
                  cv::approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                  // Square contours should have 4 vertices after approximation
                  // relatively large area (to filter out noisy contours)
                  // and be convex.
                  if( approx.size() == 4 &&
                    std::fabs(contourArea(Mat(approx))) > 500 && //was 1000
                    std::fabs(contourArea(Mat(approx))) < 0.3*image.rows*image.cols &&
                    cv::isContourConvex(Mat(approx)) ) {
                        double max_cosine = 0;
      
                        for( int j = 2; j < 5; j++ ) {
                        // Find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        max_cosine = MAX(max_cosine, cosine);
                  }

                        // If cosines of all angles are small
                        // (all angles are ~90 degree) then store 
                        if( max_cosine < 0.25 )
                              rectangles.push_back(cv::boundingRect(approx));
                  }
            }
      }
}


// The function draws all the squares in the image
void drawRectangles(Mat& image, const vector<Rect>& rectangles) {
      for( int i = 0; i < rectangles.size(); i++ ) {
            cv::rectangle(image, rectangles[i], CV_RGB(255,0,0), 1);
	    //const Point* p = &rectangles[i][0];
	    //int n = (int)rectangles[i].size();
	    //cv::polylines(image, &p, &n, 1, true, Scalar(255,0,0), 1);
      }
      showim("All rectangles detected", image);      
}


// Constructor
FindBoxByLabel::FindBoxByLabel(int argc, char *argv[]) {
	// Stores queried label
      queried_label = "";
      for (int i = 2; i < argc; i++) {
            queried_label += argv[i];
            if (i != argc - 1) queried_label += " ";
      }
      std::transform(queried_label.begin(), queried_label.end(), queried_label.begin(), ::tolower);

      input_img = imread(argv[1]);
	
      // Used for displaying images later
	group_img = input_img.clone();					
	valid_img = input_img.clone();					
	sample_img = input_img.clone();					
	match_img = input_img.clone();
      rectangle_img = input_img.clone();
      rect_display = input_img.clone();
}

inline char* FindBoxByLabel::callTesseract(Mat& input) {
      tesseract::TessBaseAPI api;

      // Initialize tesseract-ocr with English, without specifying tessdata path
      if (api.Init(NULL, "eng")) {
	    fprintf(stderr, "Could not initialize tesseract.\n");
	    exit(1);
      }

      // Open input image with leptonica library
      api.SetImage((uchar*)input.data, input.size().width, \
                   input.size().height, input.channels(), \
				   input.step1());

      // Return OCR result
      return api.GetUTF8Text();
}

// Dynamic programming implementation of the longest common substring algorithm
int FindBoxByLabel::lcSubStr(string X, string Y) { 
      int m = X.length();
      int n = Y.length();

      int lcs[m+1][n+1];
      for (int i = 0; i <= n; i++) lcs[0][i] = 0;
      for (int i = 0; i <= m; i++) lcs[i][0] = 0;

      for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                  if (X[i-1] == Y[j-1])
                        lcs[i][j] = lcs[i-1][j-1] + 1;
                  else lcs[i][j] = 0;
            }
      }

      int result = -1;
      for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++)
                  if (result < lcs[i][j]) result = lcs[i][j];
      }      
      return result;
}

// Currently not in use
int FindBoxByLabel::levenshteinDistance(const char* query, const char* test) {
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

void FindBoxByLabel::findComponentCenters() {
      // To grayscale
      cv::cvtColor(input_img, input_img, CV_RGB2GRAY);
      
      // Calculate positive, negative and total horizontal gradient
      const int KERNEL_SIZE = 1;
      Mat pos;
      cv::Sobel(input_img, pos, CV_64F, 1, 0, KERNEL_SIZE);
      Mat both = abs(pos);
      Mat neg = both - pos;

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


void FindBoxByLabel::groupConnectedComponents() {
     const int MAX_HORIZ = 7;                   // x-tolerance for grouping
     const int MAX_VERT  = 4;                   // y-tolerance for grouping
     int xdist, ydist;
     int g_id;                                  // These are used to store
     int g_id2;                                 // existing group IDs
     std::unordered_set<int> group;             // These are used to store and
     std::unordered_set<int> group2;            // merge existing groups
     bool in_a_set;
     std::unordered_set<int> all_grouped;       // Stores all CCs that have been grouped
     std::unordered_map<int,int> cc_group_ids;  // Stores the ID of the group that 
                                                // each CC is in

     // For each pair of connected components
     for (int i = 0; i < bounding_boxes.size(); i++) {
            // Not double searching a pair, as (i,j) == (j,i)
            for (int j = i+1; j < bounding_boxes.size(); j++) {
                  // If both components are in the same group already, move to next pair
                  if ( (all_grouped.find(i) != all_grouped.end()) && \
                       (all_grouped.find(j) != all_grouped.end()) && \
                       (cc_group_ids.at(i) == cc_group_ids.at(j)) ) continue;
                  else {
                        // Otherwise find the distances between the two CCs
                        xdist = std::abs(box_centers[i].x - box_centers[j].x);
                        ydist = std::abs(box_centers[i].y - box_centers[j].y);
                        // If they're within the tolerances
                        if (( xdist < MAX_HORIZ ) && ( ydist < MAX_VERT )) {
                              // If both CCs have been grouped but are in separate groups,
                              // just merge the two groups (voiding one)
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
                              // If just one of the CCs is in a group already, just add the other
                              // CC to that group
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
                              // Finally, if neither CC is in a group, create a new group and add
                              // them to it
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
     

      const float MIN_AR = 0.5;
const int num_pos = pos_contours.size();

      // Find the bounding box for each of the groupings
      // (the smallest bounding rectangle around all the CCs together)
      pos_contours.insert(pos_contours.end(), neg_contours.begin(), neg_contours.end());
int z = 0;
      for (int i = 0; i < candidate_groups.size(); i++) {
      
            // If this group got merged to another, it's no longer valid
            if (voided_groups.find(i) == voided_groups.end()) { 
                  // If there are more than 4 CCs in a group
                  if (candidate_groups[i].size() > 4) {
                        group = candidate_groups[i];
                        vector<Point> contour_group;
                        for (int id : group) {
                              vector<Point> contour = pos_contours[id];
                              contour_group.insert(contour_group.end(), contour.begin(), contour.end());
                        }
                        Rect big_box = cv::boundingRect(contour_group);
                        // Only consider this group to be a text candidate if it's small enough and
                        // if it isn't too tall and narrow
                        float ar = (float)big_box.width / (float)big_box.height;
                        float area = (float)big_box.width * (float)big_box.height;
                        if ( (ar > MIN_AR) && (area < 0.2*input_img.rows*input_img.cols) && (area > 150) ) {
                              candidate_boxes.push_back(big_box);            
                              double pos_length = 0;
                              double neg_length = 0;
                              for (int cc : candidate_groups[i]) {
                                    if (cc < num_pos) pos_length += cv::arcLength(pos_contours[cc],1);
                                    else neg_length += cv::arcLength(pos_contours[cc],1); 
                              }
                              cout << z << ":  " << pos_length << "   " << neg_length << "    " << pos_length / neg_length << endl;      
                              z++;
                        }
                  }
            }
      }
}


void FindBoxByLabel::calculateHistogramExtrema(Mat subsample) {
      static float range[] = {0, 255};
	static const float *ranges[] = {range};

	// Calculate histogram for this potential text region
	cv::calcHist(&subsample, 1, 0, Mat(), hist, 1, &hist_size, ranges, true, false);

	// Find the leftmost and right histogram peaks (which
	// correspond to the background and foreground colors)
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

	// Determine which peak is the background and which is the 
	// foreground (background has a larger histogram value)
	if (std::max(left_max, right_max) == right_max) {
		bg_color = right_ind;
		fg_color = left_ind;
	} else {
		bg_color = left_ind;
		fg_color = right_ind;
	}

      // Trying persistence1d
      vector<float> hist_vec;
      for (int i = 0; i < hist_size; i++) {
            hist_vec.push_back(hist.at<float>(i));
      }
      
      p1d::Persistence1D p;
      p.RunPersistence(hist_vec);
      vector<p1d::TPairedExtrema> extrema;
      p.GetPairedExtrema(extrema,9);

      // Display the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double)hist_w/hist_size );
	Mat hist_img(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
	normalize(hist, hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, Mat());
	for (int i = 1; i < hist_size; i++) {
		cv::line(hist_img, Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
		                   Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
		                   cv::Scalar(255,50,0), 1, 8, 0);
      }

      for(vector< p1d::TPairedExtrema >::iterator it = extrema.begin(); it != extrema.end(); it++)
	      cv::circle(hist_img, Point(bin_w*((*it).MaxIndex), hist_h - cvRound(hist.at<float>((*it).MaxIndex))), 2, cv::Scalar(0,255,0), 1);
//      cout << "Number of peaks:  " << extrema.size() << endl;
}


void FindBoxByLabel::preprocessForOCR(Mat subsample) {
	// Difference between bg and fg colors
	int color_spread = std::abs(bg_color - fg_color);

	// Creates mask where AA text is close to the text color,
	// effectively segmenting the letters
	Mat fg_mask;
	if (bg_color > fg_color) // Dark text on light
		  fg_mask = subsample < fg_color + 0.40*color_spread;
	else // Light text on dark
		  fg_mask = subsample > fg_color - 0.40*color_spread;

	// Upscales mask before sending to Tesseract
	int scale_factor = 3;
	cv::resize(fg_mask, preprocessed_text_img, cv::Size(0,0), scale_factor, scale_factor, cv::INTER_NEAREST);

}


void FindBoxByLabel::locateBestFieldMatch(int best_ind) {  
      Rect label_box = candidate_boxes[best_ind];
      Point label_center, field_center;
      label_center.x = label_box.x + label_box.width/2;
      label_center.y = label_box.y + label_box.height/2;
      int height;

      int matching_field_ind = -1;
      float x_dist;
      float y_dist;
      float best_dist = 9999999;

      for (int i = 0; i < rectangles.size(); i++) {
            field_center.x = rectangles[i].x + rectangles[i].width/2;
            field_center.y = rectangles[i].y + rectangles[i].height/2;
            height = rectangles[i].height;

            x_dist = field_center.x - label_center.x;
            y_dist = field_center.y - label_center.y;
            if (x_dist > -1*height && y_dist > -1*height) {
            if ( (rectangles[i].x > label_box.x+label_box.width) ||
                 (rectangles[i].y > label_box.y+label_box.height) ) {
                  if ( x_dist*x_dist + y_dist*y_dist < best_dist ) {
                        best_dist = x_dist*x_dist + y_dist*y_dist;
                        matching_field_ind = i;
                  }
            }
            }
      }      

      int x = rectangles[matching_field_ind].x;
      int y = rectangles[matching_field_ind].y;
      int w = rectangles[matching_field_ind].width;
      int h = rectangles[matching_field_ind].height;

      int x0, y0;
      vector<int> matched_text_inds;

      for (int i = 0; i < candidate_boxes.size(); i++) {
            x0 = candidate_boxes[i].x + candidate_boxes[i].width/2;
            y0 = candidate_boxes[i].y + candidate_boxes[i].height/2;
            if ( x < x0 && x0 < x+w && y < y0 && y0 < y+h )
                  matched_text_inds.push_back(i);
      }

      cout << "Text field corresponding to given label:" << endl;
      for (int i : matched_text_inds) {
            cout << matched_text[i] << endl;
            cv::rectangle(match_img, candidate_boxes[i], CV_RGB(255,0,0), 2);
      }
      showim("Found label and text field", match_img);
}

void FindBoxByLabel::findBoxByLabel() {
      findComponentCenters();
      groupConnectedComponents();
      
      // For each of the candidate text regions
	for (int i = 0; i < candidate_boxes.size(); i++) {
      // Sample a text region from original screen and grayscale
        Mat subsample(sample_img, candidate_boxes[i]);
              cv::cvtColor(subsample, subsample, CV_RGB2GRAY);
              calculateHistogramExtrema(subsample);

            preprocessForOCR(subsample);
            char* found_text = callTesseract(preprocessed_text_img);
            string text_str = found_text;
            std::transform(text_str.begin(), text_str.end(), text_str.begin(), ::tolower);
            matched_text.push_back(text_str);
            // Calculate the levenshtein distance between Tesseract output
		// and the user input
		// Only keep that score if it's better than all previously calculated scores
            curr_score = lcSubStr(queried_label, text_str);
		if (curr_score > best_score) {
		      best_score = curr_score;
		      best_ind = i;
		      best_match = found_text;
            }
      showim("Sub", subsample);
      waitKey();
      }


	cout << "Query label:        " << queried_label << endl;
	cout << "Matched label:   " << best_match << endl;
	
      cv::rectangle(match_img, candidate_boxes[best_ind], CV_RGB(0,255,0), 2);

	// Display bounding boxes of groups
	for (Rect big_box : candidate_boxes)
		cv::rectangle(group_img, big_box, CV_RGB(rand()%256, rand()%256, rand()%256), 2);
	showim("Candidate groups", group_img);

      preprocessImgForRectFinder(rectangle_img);
      findRectangles(rectangle_img, rectangles);
//      drawRectangles(rect_display, rectangles);

}
