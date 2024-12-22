/* 背景差分法
将视频序列中的背景和前景分离的技术。通过与背景图像的差异来检测前景物体

*/

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);  // 打开摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 创建背景减法器
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();

    cv::Mat frame, fgMask;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 应用背景减法
        pBackSub->apply(frame, fgMask);

        // 显示结果
        cv::imshow("Frame", frame);
        cv::imshow("Foreground Mask", fgMask);

        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
