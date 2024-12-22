#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// 模板匹配函数
Point matchTemplateInImage(const Mat& image, const Mat& templ, double& maxVal) {
    Mat result;
    // 检查并转换图像为灰度图像
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    if (templ.channels() == 3) {
        cv::cvtColor(templ, templ, cv::COLOR_BGR2GRAY);
    }

    // 确保它们都是 CV_8U
    if (image.depth() != CV_8U) {
        image.convertTo(image, CV_8U);
    }
    if (templ.depth() != CV_8U) {
        templ.convertTo(templ, CV_8U);
    }

    matchTemplate(image, templ, result, TM_CCOEFF_NORMED);
    
    // 找到匹配位置
    double minVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    return maxLoc;
}

int main() {
    // 载入模板图像
    Mat templ = imread("/home/wys/Desktop/Project/VisionProject/Opencv/images/origin/5.png", IMREAD_GRAYSCALE);
    if (templ.empty()) {
        cout << "Error: Unable to load template image" << endl;
        return -1;
    }
    cv::resize(templ, templ, Size(640, 480));  // 缩放模板图像，以便匹配速度更快
    // 载入视频流
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        cout << "Error: Unable to open video capture device" << endl;
        return -1;
    }
    Mat frame, grayFrame;
    while (true) {
        cap >> frame;  // 获取帧
        cv::resize(frame, frame, Size(640, 480));

        if (frame.empty()) {
            cout << "Error: Frame capture failed" << endl;
            break;
        }

        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // 模板匹配
        double maxVal;
        Point matchLoc = matchTemplateInImage(grayFrame, templ, maxVal);

        // 如果匹配结果好于阈值，绘制匹配区域
        if (maxVal > 0.7) {  // 设置阈值为0.7，可以根据需要调整
            rectangle(frame, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2);
        }

        imshow("Template Matching", frame);
        if (waitKey(30) >= 0) break;  // 按键退出
    }

    return 0;
}
