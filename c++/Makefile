CC=g++ -std=c++11
CFLAGS=-c -Wall
INCLUDES=-I/usr/include/leptonica -I/usr/include/tesseract/
LDFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract
SOURCES=boxFinder.cpp findRectangles.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=textbox

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(INCLUDES) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

