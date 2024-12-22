#include <opencv2/opencv.hpp>
#include <iostream>

void detectChannelType(const cv::Mat& image) {
    // 获取图像的类型和通道数
    int type = image.type();
    int channels = image.channels(); // 获取通道数

    // 打印图像类型和通道数
    std::cout << "Image type: " << type << std::endl;
    std::cout << "Number of channels: " << channels << std::endl;

    // 根据类型显示更易读的格式
    switch (type) {
        case CV_8UC1: std::cout << "Type: CV_8UC1 (8-bit single channel)" << std::endl; break;
        case CV_8UC3: std::cout << "Type: CV_8UC3 (8-bit 3 channels)" << std::endl; break;
        case CV_16UC1: std::cout << "Type: CV_16UC1 (16-bit single channel)" << std::endl; break;
        case CV_16UC3: std::cout << "Type: CV_16UC3 (16-bit 3 channels)" << std::endl; break;
        case CV_32FC1: std::cout << "Type: CV_32FC1 (32-bit float single channel)" << std::endl; break;
        case CV_32FC3: std::cout << "Type: CV_32FC3 (32-bit float 3 channels)" << std::endl; break;
        // 你可以根据需要添加其他类型
        default: std::cout << "Type: Unknown" << std::endl; break;
    }

    // 检查颜色通道顺序（BGR 或 RGB）
    if (channels == 3) {
        // 读取第一个像素点的 BGR 值
        cv::Vec3b pixel = image.at<cv::Vec3b>(0, 0);
        std::cout << "First pixel (BGR): " << "B: " << (int)pixel[0] << ", G: " << (int)pixel[1] << ", R: " << (int)pixel[2] << std::endl;
        std::cout << "Color order: BGR" << std::endl;

        // 将 BGR 转换为 RGB 和 HSV
        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
        cv::Mat hsvImage;
        cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

        // 检查 RGB 的第一个像素
        cv::Vec3b pixelRGB = rgbImage.at<cv::Vec3b>(0, 0);
        std::cout << "First pixel (RGB): " << "R: " << (int)pixelRGB[0] << ", G: " << (int)pixelRGB[1] << ", B: " << (int)pixelRGB[2] << std::endl;

        // 检查 HSV 的第一个像素
        cv::Vec3b pixelHSV = hsvImage.at<cv::Vec3b>(0, 0);
        std::cout << "First pixel (HSV): " << "H: " << (int)pixelHSV[0] << ", S: " << (int)pixelHSV[1] << ", V: " << (int)pixelHSV[2] << std::endl;

        std::cout << "Color order: BGR" << std::endl;
    } else if (channels == 1) {
        std::cout << "Image is single-channel (grayscale)" << std::endl;
    } else {
        std::cout << "Unsupported channel count" << std::endl;
    }
}

int main() {
    // 创建视频捕捉对象
    cv::VideoCapture cap(2); // 0 表示默认摄像头

    // 检查摄像头是否打开
    if (!cap.isOpened()) {
        std::cerr << "Could not open the camera!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        // 从摄像头读取图像
        cap >> frame;

        // 检查图像是否成功读取
        if (frame.empty()) {
            std::cerr << "Could not read frame from camera!" << std::endl;
            break;
        }

        // 调用函数检测通道类型
        detectChannelType(frame);

        // 显示图像
        cv::imshow("Camera Frame", frame);

        // 按下 'q' 键退出
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // 释放摄像头
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
