#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

int main() {
    cv::CascadeClassifier handCascade;
    handCascade.load("/home/wys/Desktop/Project/VisionProject/Opencv/lbpcascade_frontalface_improved.xml"); // 加载训练好的手势检测模型

    cv::VideoCapture cap(0); // 从摄像头捕获视频流
    cv::Mat frame;

    while (true) {
        cap >> frame; // 读取每一帧
        std::vector<cv::Rect> hands;
        handCascade.detectMultiScale(frame, hands, 1.1, 3, 0, cv::Size(30, 30)); // 检测手势

        for (const auto& hand : hands) {
            cv::rectangle(frame, hand, cv::Scalar(255, 0, 0), 2); // 绘制检测到的手势
        }

        cv::imshow("Hand Gesture Detection", frame); // 显示结果

        if (cv::waitKey(30) >= 0) break; // 按任意键退出
    }

    return 0;
}
