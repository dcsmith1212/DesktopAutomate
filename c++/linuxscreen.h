#ifndef LINUXSCREEN_H
#define LINUXSCREEN_H

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdint>
#include <cstring>
#include <vector>

void ImageFromDisplay(std::vector<uint8_t>&, int&, int&, int&);

#endif
