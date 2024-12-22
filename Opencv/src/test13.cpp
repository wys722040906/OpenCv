#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

int main() {
    // 打开视频捕获设备或视频文件
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video stream or file" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    cap >> frame; // 读取第一帧

    if (frame.empty()) {
        std::cerr << "Error: First frame is empty!" << std::endl;
        return -1;
    }

    std::vector<cv::Ptr<cv::Tracker>> trackers;  // 存储多个跟踪器对象
    std::vector<cv::Rect> bboxes;  // 存储多个目标区域
    
    // 手动选择多个跟踪区域（矩形框）
    bool selecting = true;
    while (selecting) {

        cv::Rect bbox = cv::selectROI("Select Object", frame, false);
        
        // 检查选择的 ROI 是否有效
        if (bbox.width == 0 || bbox.height == 0) {
            std::cerr << "Invalid ROI selected, exiting selection mode." << std::endl;
            break;
        }

        // 创建新的 KCF 跟踪器并初始化
        cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
        tracker->init(frame, bbox);
        
        // 将跟踪器和矩形框保存到容器中
        trackers.push_back(tracker);
        bboxes.push_back(bbox);
        
        // // 重新绘制当前帧上所有已选择的 ROI
        for (size_t i = 0; i < bboxes.size(); ++i) {
            cv::rectangle(frame, bboxes[i], cv::Scalar(255, 0, 0), 2, 1);
        }

        std::cout << "Press 'n' to select another object, or any other key to start tracking." << std::endl;
        //阻塞进行
        if (cv::waitKey(0) != 'n') {
            selecting = false;
        }
    }

    // 开始跟踪
    while (cap.read(frame)) {
        // 遍历每个跟踪器，更新每个目标的位置
        for (size_t i = 0; i < trackers.size(); i++) {
            bool ok = trackers[i]->update(frame, bboxes[i]);
            
            if (ok) {
                // 如果跟踪成功，绘制边界框
                cv::rectangle(frame, bboxes[i], cv::Scalar(255, 0, 0), 2, 1);
            } else {
                std::cout << "Tracking failed for object " << i + 1 << std::endl;
            }
        }
        
        // 显示跟踪结果
        cv::imshow("Multi-Object Tracking", frame);
        if (cv::waitKey(30) == 27) break; // 按下 ESC 键退出
    }

    return 0;
}
