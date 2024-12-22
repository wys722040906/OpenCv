#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    // 创建 HOG(方向梯度直方图--一种特征描述符)描述符
    cv::HOGDescriptor hog;
    //hog使用一个预定义的支持向量机（SVM）检测器
    //内置的默认行人检测器(静态方法)
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());//

    // 打开视频文件或摄像头
    cv::VideoCapture cap(0); // 使用摄像头
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // 读取每一帧
        if (frame.empty()) break;

        // 检测行人
        std::vector<cv::Rect> detections;
        //给定图像帧检测多个物体--多尺度检测
        hog.detectMultiScale(frame, detections, 0, cv::Size(8, 8), cv::Size(0, 0), 1.05, 2);

        // 绘制检测到的行人
        for (const auto& rect : detections) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2); // 用绿色矩形标记行人
        }

        // 显示结果
        cv::imshow("HOG + SVM Pedestrian Detection", frame);

        // 按 'q' 键退出
        if (cv::waitKey(30) == 'q') break;
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
