#include <opencv2/opencv.hpp>
#include <iostream>

// #include <opencv2/opencv.hpp>
// #include <iostream>
#include <chrono>

int main() {
    cv::VideoCapture cap(0);  // 打开摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

//cv
    double fps_cv = 0.0;
    int64 start_cv = cv::getTickCount();
    cv::Mat frame;
//chrono
    double fps = 0.0;
    auto start = std::chrono::high_resolution_clock::now();


    while (cap.read(frame)) {
        // 处理图像（例如灰度转换）
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);


        // cv
        int64 end_cv = cv::getTickCount();
        double seconds_cv = (end_cv - start_cv) / cv::getTickFrequency();
        fps_cv = 1.0 / seconds_cv;

        //chrono
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        fps = 1.0 / elapsed.count();     


        // 输出帧率
        std::cout << "FPS_CV: " << static_cast<int>( fps_cv) << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        // 更新计时器
        start_cv = end_cv;
        start = end;


        cv::imshow("Frame", frame);
        if (cv::waitKey(10) == 27) break;  // 按ESC退出
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
