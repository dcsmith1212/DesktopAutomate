#include "textboxfinder.h"

char* callTesseract(cv::Mat inputImg) {
        char *outText;

        tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
        // Initialize tesseract-ocr with English, without specifying tessdata path
        if (api->Init(NULL, "eng")) {
                fprintf(stderr, "Could not initialize tesseract.\n");
                exit(1);
        }


        // Open input image with leptonica library
        api->SetImage((uchar*)inputImg.data, inputImg.size().width, inputImg.size().height, inputImg.channels(), inputImg.step1());
        // Get OCR result
        outText = api->GetUTF8Text();
        //printf("OCR output:\n%s", outText);

        return outText;
        // Destroy used object and release memory
        //api->End();
        //delete [] outText;
}

