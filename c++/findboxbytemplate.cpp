#include "findboxbytemplate.h"
#include "ocvmacros.h"

#include <vector>
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;

using cv::Point;
using cv::Rect;
using cv::Mat;
using cv::pyrDown;
using cv::pyrUp;
using cv::Size;
using cv::Scalar;
using cv::imread;

using cv::namedWindow;
using cv::waitKey;

// Blank constructor -- automatically calls for user input upon initialization
FindBoxByTemplate::FindBoxByTemplate() {
      getInputImages();
}

// Param. constructor -- takes predefined screen and template images
FindBoxByTemplate::FindBoxByTemplate(Mat& input_screen, Mat& input_template) {
      fullscreen_img = input_screen;
      template_img   = input_template;
}

// Helper function for findSquares:
// Finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double FindBoxByTemplate::angle( Point pt1, Point pt2, Point pt0 ) {
      double dx1 = pt1.x - pt0.x;
      double dy1 = pt1.y - pt0.y;
      double dx2 = pt2.x - pt0.x;
      double dy2 = pt2.y - pt0.y;
      return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// Returns sequence of squares detected on the image.
void FindBoxByTemplate::findRectangles(const Mat& image, vector<vector<Point> >& rectangles, vector<double>& max_cosines) {
      rectangles.clear();

      Mat bin;
      vector<vector<Point> > contours;

      // Find rectangles for a series of low threshold values
      for (int thresh = 5; thresh <= 75; thresh += 10) {
            bin = image >= thresh;
            
            // Find contours and store them all as a list
            cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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
                              rectangles.push_back(approx);
                              max_cosines.push_back(max_cosine);
                  }
            }
      }
}

// The function draws all the squares in the image
void FindBoxByTemplate::drawRectangles(Mat& image, const vector<vector<Point>>& rectangles) {
      for( int i = 0; i < rectangles.size(); i++ ) {
	    const Point* p = &rectangles[i][0];
	    int n = (int)rectangles[i].size();
	    cv::polylines(image, &p, &n, 1, true, Scalar(255,0,0), 1);
      }
      showim("All rectangles detected", image);      
}

void FindBoxByTemplate::preprocessImage() {
      // Copy input image for sampling later in decompose_image
      rect_display_img = found_img.clone();
      field_parent_img = found_img.clone();

      //Converts input image to grayscale
      Mat gray;
      cvtColor(found_img, gray, CV_BGR2GRAY);

      // Will be the "whitened out" version of the found template image`
      gray_clone_img = gray.clone();

      // Converts lap and gray to same type, then find Laplacian of query image
      Mat lap;
      lap.convertTo(lap, CV_8UC1);
      gray.convertTo(gray, CV_8UC1);
      cv::Laplacian(gray,lap,0);	

      processed_img = lap;
}

void FindBoxByTemplate::decomposeImage() {
      // Find rectangle that makes up boundary of text field
      vector<vector<Point>> rectangles;
      vector<double> max_cosines;
      findRectangles(processed_img, rectangles, max_cosines);

      // Number of potential text fields
      cout << "# of potential text fields: " << rectangles.size() << endl;
      
      if (rectangles.size() == 0)
            cout << "No rectangles found!  Seg fault coming...." << endl << endl;

      drawRectangles(rect_display_img, rectangles);

      // Sample the found image further, leaving only the text field
      double min_cosine = max_cosines[0];
      int min_index = 0;
      for (int i = 1; i < rectangles.size(); i++) {
            if (max_cosines[i] < min_cosine) {
                  min_cosine = max_cosines[i];
                  min_index = i;
            }
      }      

      Rect field_bbox = cv::boundingRect(rectangles[min_index]);
      Mat field(field_parent_img, field_bbox);
      text_field_img = field;

	// "White out" of textfield to leave only label
      int nrows = field_bbox.height;
	int ncols = field_bbox.width;
	Mat blank(nrows, ncols, CV_8U, cv::Scalar(255));
	Mat roi(gray_clone_img(field_bbox));
	blank.copyTo(roi);
	text_label_img = gray_clone_img;
}

void FindBoxByTemplate::enhanceImage(Mat& input_img) {
      // Upscales the image and blurs to give a percieved increase in resolution
      Mat enhanced_img;
      cv::resize(input_img, input_img, cv::Size(), 5, 5);      
      cv::GaussianBlur(input_img, enhanced_img, cv::Size(0, 0), 3);
      cv::addWeighted(input_img, 1.5, enhanced_img, -0.5, 0, input_img);
}

void FindBoxByTemplate::findTemplateImage() {
      // Runs NCC on template and fullscreen
      Mat output;
      cv::matchTemplate(fullscreen_img, template_img, output, CV_TM_CCOEFF_NORMED);

      // Finds best match point
      double min, max;
      Point min_loc, max_loc;
      cv::minMaxLoc(output, &min, &max, &min_loc, &max_loc);

      // For rectangle definition
      Point bottom_right;
      bottom_right.x = max_loc.x + template_img.cols;
      bottom_right.y = max_loc.y + template_img.rows;
      
      // Used to sample fullscreen and to store template stats
      Rect subset(max_loc, bottom_right);
      template_stats = subset;

      // Displays rectangle around found template area
      Mat draw_img = fullscreen_img.clone();
      cv::rectangle(draw_img, max_loc, bottom_right, CV_RGB(255,0,0));
      showim("Template Location", draw_img);
      
      // Samples the screenshot with found template image
      cv::Mat sampled_template(fullscreen_img, subset);
      found_img = sampled_template;
}

void FindBoxByTemplate::getInputImages() {
      cout << "\n-----------------------------------\n"
	  "Textbox Location and Data Retrieval\n"
	  "   Implementation as of 9/16/16    \n"
	  "-----------------------------------\n\n"
	  "Which image pair would you like to test?\n"
	  " (enter name of one of these directories):\n\n";

      // Shows current set of screenshot-template pairs to choose from
      system("ls images/textboxes/");
      cout << ">> ";

      string test_id;
      cin >> test_id;

      cout << "images/textboxes/" + test_id + "/screen.png" << endl;

      // Reads the two images into memory
      fullscreen_img = imread("images/textboxes/" + test_id + "/screen.png", CV_LOAD_IMAGE_COLOR);
      template_img = imread("images/textboxes/" + test_id + "/template.png", CV_LOAD_IMAGE_COLOR);	
}

// Not yet incorporated into main program
void FindBoxByTemplate::imageFromDisplay(vector<uint8_t>& pixels, int& width, int& height, int& bits_per_pixel) {
      Display* display = XOpenDisplay(nullptr);
      Window root = DefaultRootWindow(display);

      XWindowAttributes attributes = {0};
      XGetWindowAttributes(display, root, &attributes);

      // Screen width and height
      width = attributes.width;
      height = attributes.height;

      XImage* img = XGetImage(display, root, 0, 0, width, height, AllPlanes, ZPixmap);
      bits_per_pixel = img->bits_per_pixel;
      pixels.resize(width * height * 4);

      memcpy(&pixels[0], img->data, pixels.size());

      XDestroyImage(img);
      XCloseDisplay(display);
}

char* FindBoxByTemplate::callTesseract(Mat input_image) {
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

void FindBoxByTemplate::findBoxByTemplate() {
      // Display chosen images
      showim("Fullscreen", fullscreen_img);
      showim("Template Image", template_img);

      // Runs NCC on query image in screenshot
      findTemplateImage();	

      // Processes the found image and then splits into textbox field 
      // and textbox label
      preprocessImage();
      decomposeImage();

      // Gives both images a percieved increase in resolution by
      //rescaling and blurring
      enhanceImage(text_field_img);
      enhanceImage(text_label_img);

      // Run the enhanced images through Tesseract
      // Store results in public variables
      text_label = callTesseract(text_label_img);
      text_field = callTesseract(text_field_img);
}
