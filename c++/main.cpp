#include <iostream>
#include <string>
#include "textboxfinder.h"
#include "ocvmacros.h"

#include <cmath>

// Standard functions/datatypes
using std::cout;
using std::endl;
using std::string;
using std::vector;

// OpenCV functions/datatypes
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::imread;

using cv::Range;

int main() {
// This is code for template-based matching
/*
      // Initialize finder object (empty constuctor will call getInputImages() automatically)
	TextBoxFinder finder;

	// Retrieves location of template in screenshot, and determines label and field text
	finder.findBoxByTemplate();
      cout << "Template location (UL corner):   (" << finder.template_stats.x << "," << finder.template_stats.y << ")" << endl;
      cout << "Template size:   (" << finder.template_stats.width << "x" << finder.template_stats.height << ")" << endl << endl;

      cout << "Tesseract output:" << endl;
      cout << "Textbox label:   " << finder.text_label << endl;
      cout << "Textbox field:   " << finder.text_field << endl;
*/


      Mat base_img = cv::imread("images/textboxes/test11/screen.png");
      showim("test", base_img);      

      int w = base_img.cols;
      int h = base_img.rows;

      int w_mid = floor(w/2);
      int h_mid = floor(h/2);


      Mat ul_img(base_img, Range(0,h_mid), Range(0,w_mid));
      Mat ur_img(base_img, Range(0,h_mid), Range(w_mid+1,w));      
      Mat ll_img(base_img, Range(h_mid+1,h), Range(0,w_mid));
      Mat lr_img(base_img, Range(h_mid+1,h), Range(w_mid+1,w));

      showim("ul",ul_img);
      showim("ur",ur_img);
      showim("ll",ll_img);
      showim("lr",lr_img);

 
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
