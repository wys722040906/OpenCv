#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 全局变量
int lower_r = 0, lower_g = 0, lower_b = 0; // RGB 通道的下限
int upper_r = 255, upper_g = 255, upper_b = 255; // RGB 通道的上限

// 形态学操作迭代次数
int ErodeIter = 1; // 默认腐蚀迭代次数
int DilateIter = 1; // 默认膨胀迭代次数
int OpenIter = 1; // 默认开运算迭代次数
int CloseIter = 1; // 默认闭运算迭代次数

// 定义一个结构体用于保存阈值设置
struct Thresholds {
    std::string detect_color = "GREEN"; // 检测颜色
    int rgb_thresold = 50;              // RGB 阈值
    int gray_thresold = 100;            // 灰度阈值
} m_all_thresold;

// 处理图像 - 特征提取和融合
void img_pretreat(const Mat &src, Mat &processed_image) {
    Mat split_image[3];
    Mat rgb_image, gray_image;

    // 分离 BGR 通道
    cv::split(src, split_image);

    // 通道减法保留颜色特征
    if (m_all_thresold.detect_color == "GREEN") {
        rgb_image = split_image[1] - split_image[0]; // G - B
    } else if (m_all_thresold.detect_color == "RED") {
        rgb_image = split_image[2] - split_image[0]; // R - B
    } else if (m_all_thresold.detect_color == "BLUE") {
        rgb_image = split_image[0] - split_image[2]; // B - R
    } else {
        rgb_image = split_image[2] - split_image[0]; // R - B（默认情况）
    }
    imshow("RGB Image", rgb_image);

    // 应用 RGB 阈值
    cv::threshold(rgb_image, rgb_image, m_all_thresold.rgb_thresold, 255, THRESH_BINARY);

    // 转换到灰度图像
    cv::cvtColor(src, gray_image, COLOR_BGR2GRAY);

    // 应用灰度阈值
    cv::threshold(gray_image, gray_image, m_all_thresold.gray_thresold, 255, THRESH_BINARY);

    // 融合两个特征
    cv::bitwise_and(gray_image, rgb_image, processed_image);

    // 形态学操作
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); // 结构元素
    // 开运算（先腐蚀后膨胀）
    for (int i = 0; i < OpenIter; i++) {
        cv::morphologyEx(processed_image, processed_image, MORPH_OPEN, element);
    }
    // 闭运算（先膨胀后腐蚀）
    for (int i = 0; i < CloseIter; i++) {
        cv::morphologyEx(processed_image, processed_image, MORPH_CLOSE, element);
    }
    // 腐蚀
    for (int i = 0; i < ErodeIter; i++) {
        cv::erode(processed_image, processed_image, element);
    }
    // 膨胀
    for (int i = 0; i < DilateIter; i++) {
        cv::dilate(processed_image, processed_image, element);
    }
}

void find_contours(const Mat &imgCanny, vector<vector<Point>> &contours, vector<Vec4i> &hierarchy, int mode, int method, const Scalar &RecColor, const int RecAreaThread) {
    cv::findContours(imgCanny, contours, hierarchy, mode, method);

    // 轮廓处理
    for (int i = 0; i < contours.size(); i++) {
        // 面积过滤
        if (cv::contourArea(contours[i]) > RecAreaThread) {
            cv::drawContours(imgCanny, contours, i, RecColor, 2, 8, hierarchy, 0); // 绘制轮廓
        } else {
            contours.erase(contours.begin() + i); // 删除面积小于阈值的轮廓
            i--;
        }
    }
}

int main() {
    VideoCapture cap(2); // 从摄像头读取图像，0表示默认摄像头
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }
    namedWindow("Live Video", WINDOW_AUTOSIZE);
    namedWindow("Processed", WINDOW_AUTOSIZE);

    // 创建滑动条（保留有效的滑动条）
    createTrackbar("RGB Threshold", "Live Video", &m_all_thresold.rgb_thresold, 255); // RGB 阈值
    createTrackbar("Gray Threshold", "Live Video", &m_all_thresold.gray_thresold, 255); // 灰度阈值

    createTrackbar("Erode Iter", "Live Video", &ErodeIter, 10);
    createTrackbar("Dilate Iter", "Live Video", &DilateIter, 10);
    createTrackbar("Open Iter", "Live Video", &OpenIter, 10);
    createTrackbar("Close Iter", "Live Video", &CloseIter, 10);

    Mat processed_image;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int mode = cv::RETR_EXTERNAL;
    int method = cv::CHAIN_APPROX_SIMPLE;
    Scalar RecColor(0, 255, 0);
    int RecAreaThread = 1000;  // 轮廓面积阈值

    while (true) {
        Mat frame;
        cap >> frame; // 读取视频帧
        if (frame.empty()) {
            cout << "Error: empty frame" << endl;
            break;
        }
        resize(frame, frame, Size(640, 480)); // 调整帧大小

        img_pretreat(frame, processed_image); // 处理图像

        find_contours(processed_image, contours, hierarchy, mode, method, RecColor, RecAreaThread);

        imshow("Live Video", frame);
        imshow("Processed", processed_image);
        if (waitKey(1) == 27) {
            break; // 按下ESC键退出
        }
    }
    return 0;
}
