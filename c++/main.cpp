#include <iostream>
#include <string>
#include "findboxbytemplate.h"
#include "textdetect.h"
#include "ocvmacros.h"

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

      // Detect text in the image
      DetectText dt(argv);      
      dt.detectText();

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
