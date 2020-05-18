#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    cv::Mat img;
    cv::Mat gauFtr;
    cv::Mat downSampledImg;
    img = cv::imread("tomato1.jpeg", 1);
    cv::GaussianBlur(img, gauFtr, cv::Size(5, 5), 0, 0);
    cv::pyrDown(gauFtr, downSampledImg);
    cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);
    cv::imshow("frame", img);
    cv::waitKey(0);
    cv::imshow("frame", gauFtr);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}