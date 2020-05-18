#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    cv::Mat img;
    cv::Mat edgeImg;
    img = cv::imread("tomato1.jpeg", 1);
    cv::Canny(img, edgeImg, 50, 150);
    cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);
    cv::imshow("frame", img);
    cv::waitKey(0);
    cv::imshow("frame", edgeImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}