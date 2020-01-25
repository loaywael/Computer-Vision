#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

    cv:: Mat img, channels[3], mergedImg;

    img = cv::imread("tomato1.jpeg", cv::IMREAD_COLOR);
    cv::split(img, channels);

    cv::namedWindow("blue-channel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("green-channel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("red-channel", cv::WINDOW_AUTOSIZE);

    cv::moveWindow("blue-channel", 500, 350);
    cv::moveWindow("green-channel", img.cols + 600, 350);
    cv::moveWindow("red-channel", img.cols * 2 + 700, 350);

    cv::imshow("blue-channel", channels[0]);
    cv::imshow("green-channel", channels[1]);
    cv::imshow("red-channel", channels[2]);
    cv::waitKey(0);

    channels[1] = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::merge(channels, 3, mergedImg);
    cv::imshow("merged-channels", mergedImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
    cv::GaussianBlur

    return 0;
}