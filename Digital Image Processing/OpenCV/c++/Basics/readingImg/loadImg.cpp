#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

    cv::Mat img = cv::imread("face0.jpg", cv::IMREAD_COLOR);
    cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);
    cv::imshow("frame", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}