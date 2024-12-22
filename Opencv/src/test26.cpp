
// 差分法： 一种简单的、低精度的“梯度检测”方法，它不涉及梯度方向等信息
//          梯度检测通常涉及卷积运算（如 Sobel、Laplacian）  差分法则直接计算差值
//          常用于运动检测、前景背景分割
// 梯度检测: 更适合边缘检测和轮廓提取


// 帧差法（运动检测）
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);  // 打开摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    cv::Mat prevFrame, currFrame, diffFrame, grayDiff;
    
    // 读取第一帧作为参考帧
    cap >> prevFrame;
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    while (true) {
        // 读取当前帧
        cap >> currFrame;
        if (currFrame.empty()) break;

        // 转换为灰度图
        cv::cvtColor(currFrame, currFrame, cv::COLOR_BGR2GRAY);

        // 计算当前帧与前一帧的差分
        cv::absdiff(prevFrame, currFrame, diffFrame);

        // 对差分图像进行阈值化以突出运动区域
        cv::threshold(diffFrame, grayDiff, 30, 255, cv::THRESH_BINARY);

        // 显示结果
        cv::imshow("Difference", grayDiff);
        if (cv::waitKey(30) >= 0) break;

        // 更新前一帧
        prevFrame = currFrame.clone();
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
