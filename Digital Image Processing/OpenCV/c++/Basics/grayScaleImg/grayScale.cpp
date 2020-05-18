#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::Mat img, grayImg;
    img = cv::imread("./tomato1.jpeg", cv::IMREAD_COLOR);
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    cv::imshow("frame", img);
    cv::waitKey(0);
    cv::imshow("frame", grayImg);
    cv::waitKey(0);
    cv::imwrite("./graImg.jpg", grayImg);
    cv::destroyAllWindows();
    return 0;
}