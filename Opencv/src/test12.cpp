#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

int main() {
    // 创建一个 KCF 跟踪器对象
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    
    // 读取视频
    cv::VideoCapture cap(2);
    cv::Mat frame;
    
    // 从视频中读取第一帧
    cap >> frame;
    
    // 手动选择跟踪的目标区域（矩形框）
    cv::Rect bbox = cv::selectROI(frame, false);
    if (frame.empty()) {
    std::cerr << "Frame is empty!" << std::endl;
    return -1;
    }
    if (bbox.width == 0 || bbox.height == 0) {
    std::cerr << "Error: Invalid ROI selected!" << std::endl;
    return -1;
    }
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // 初始化跟踪器
    tracker->init(frame, bbox);

    while (cap.read(frame)) {
        // 更新跟踪器
        bool ok = tracker->update(frame, bbox);
        
        if (ok) {
            // 如果跟踪成功，绘制边界框
            cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
        } else {
            std::cout << "跟踪失败！" << std::endl;
        }
        
        // 显示结果
        cv::imshow("Tracking", frame);
        if (cv::waitKey(30) == 27) break; // 按下ESC退出
    }
    return 0;
}
