#include "ocvmacros.h"

void showim(std::string windowName, cv::Mat image) {
	cv::namedWindow(windowName, 0);
	cv::imshow(windowName, image);
}

// tic and toc matlab equivalents
std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}
