#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {

    cv::Mat frame;
    cv::VideoCapture cap(0);
    while (cap.isOpened()) {
        cap >> frame;
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}