#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    cv::Mat img, dimmedImg;

    img = cv::imread("./tomato1.jpeg", cv::IMREAD_COLOR);
    dimmedImg = img.clone();
    auto end = dimmedImg.end<cv::Vec3b>();
    for (auto it = dimmedImg.begin<cv::Vec3b>(); it != end; it++) {
        (*it)[0] *= 0.5;
        (*it)[1] *= 0.5;
        (*it)[2] *= 0.5;
    }
    
    cv::namedWindow("source", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("dimmed", cv::WINDOW_AUTOSIZE);
    
    cv::moveWindow("source", 750, 350);
    cv::imshow("source", img);

    cv::moveWindow("dimmed", 750 + img.cols + 50, 350);
    cv::imshow("dimmed", dimmedImg);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}