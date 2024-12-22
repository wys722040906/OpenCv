#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>

int main() {
    // 光流法参数
    cv::Size winSize(15, 15); // 窗口大小
    int maxLevel = 2; // 最大搜索层数
    //type--v::TermCriteria::COUNT(迭代次数) | cv::TermCriteria::EPS(精度)       maxCount--最大迭代次数  EPS--参数变化的最小精度

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03); // 停止条件

    // 初始化目标位置和特征点
    cv::Rect2d bbox;
    std::vector<cv::Point2f> prev_points;
    cv::Mat prev_gray;

    // 创建跟踪器
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    // 读取视频文件
    // cv::VideoCapture cap("/home/wys/Desktop/Project/VisionProject/Opencv/images/GreebRabbish_2.mp4");
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return -1;
    }

    // 读取第一帧图像
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "无法读取视频帧" << std::endl;
        return -1;
    }

    // 创建一个可以调整大小的窗口
    cv::namedWindow("选择目标", cv::WINDOW_NORMAL);

    // 在第一帧图像中选择目标区域
    bbox = cv::selectROI("选择目标", frame, false, true);
    cv::destroyWindow("选择目标");

    // 初始化跟踪器
    tracker->init(frame, bbox);
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);

    while (true) {
        // 读取当前帧图像
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // 使用光流法跟踪特征点
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (prev_points.empty()) {
            // 在第一帧中计算特征点
            cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
            cv::rectangle(mask, bbox, cv::Scalar(255), -1);
            cv::goodFeaturesToTrack(gray, prev_points, 100, 0.3, 7, mask, 7);
        }

        if (!prev_points.empty() && !prev_gray.empty()) {
            // 计算光流
            std::vector<cv::Point2f> next_points;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_points, next_points, status, err, winSize, maxLevel, criteria);

            // 选取符合条件的特征点和目标位置
            std::vector<cv::Point2f> good_points;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) {
                    good_points.push_back(next_points[i]);
                }
            }

            if (!good_points.empty()) {
                // 计算新的边界框
                cv::Rect2d new_bbox = cv::boundingRect(good_points);
                bbox = new_bbox;

                // 更新目标位置和特征点
                prev_points = good_points;

                // 绘制矩形框和特征点
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
                for (const auto& point : good_points) {
                    cv::circle(frame, point, 3, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        // 显示图像
        cv::imshow("Frame", frame);

        // 按下Esc键退出
        if (cv::waitKey(1) == 27) {
            break;
        }

        // 更新前一帧图像和特征点
        prev_gray = gray.clone();
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
