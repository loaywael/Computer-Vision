#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {

    cv::Mat frame;
    cv::VideoCapture capture;
    capture.open("lanePath.avi");
    while (capture.isOpened()) {
        capture >> frame;
        if (frame.empty() || cv::waitKey(1) == 'q') {
            break;
        }
        cv::imshow("frame", frame);
    }
    return 0;
}