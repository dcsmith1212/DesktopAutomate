# This target will compile all files
CC=g++ -std=c++11
#SOURCES=decomposeimage.cpp findquery.cpp linuxinterface.cpp ocvmacros.cpp textboxmain.cpp enhanceimage.cpp findrectangles.cpp linuxscreen.cpp tessocr.cpp
SOURCES=textboxfinder.cpp ocvmacros.cpp main.cpp
INCLUDES=-I/usr/include/leptonica -I/usr/include/tesseract
LDFLAGS=-I/usr/include/leptonica -I/usr/include/tesseract -I/usr/include/opencv -I/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract -lX11
EXECUTABLE=textboxfinder

all:
	$(CC) -o $(EXECUTABLE) $(SOURCES) $(INCLUDES) $(LDFLAGS) 

clean:
	rm -f $(EXECUTABLE) quad

#CC=g++ -std=c++11
#CFLAGS=-Wall
#LDPATH=/home/derek/develop/libs/notmine/c++/opencv/opencv4android/sdk/native/libs/x86_64
#LDFLAGS=-I/usr/include/leptonica -I/usr/include/tesseract -I/usr/include/opencv -I/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -llept -ltesseract -lX11
#SOURCES=decomposeimage.cpp findquery.cpp linuxinterface.cpp ocvmacros.cpp textboxmain.cpp enhanceimage.cpp findrectangles.cpp linuxscreen.cpp tessocr.cpp
#OBJECTS=$(SOURCES:.cpp=.o)
#EXECUTABLE=textboxfinder
#
#all: $(EXECUTABLE)
#
#$(EXECUTABLE): $(OBJECTS)
#	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@
#
#%.o: %.cpp
#	$(CC) $(CFLAGS) -c $< -o $@
#
#clean:
#	rm -f *o $(EXECUTABLE)
