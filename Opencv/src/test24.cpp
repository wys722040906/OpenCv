// #include <iostream>
// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// int main() {
//     // 打开摄像头
//     VideoCapture cap(0); // 使用默认摄像头，修改为其他索引以选择其他摄像头
//     if (!cap.isOpened()) {
//         cerr << "Error: Could not open camera!" << endl;
//         return 1;
//     }

//     // 创建 QRCodeDetector 对象
//     QRCodeDetector qrDecoder;

//     while (true) {
//         Mat frame;
//         cap >> frame; // 从摄像头捕捉帧
//         if (frame.empty()) {
//             cerr << "Error: Could not grab a frame!" << endl;
//             break;
//         }

//         // 解码二维码
//         string decodedData;
//         Mat points; // 用于存储二维码的四个角点
//         decodedData = qrDecoder(frame, points);

//         // 如果检测到二维码，则绘制框和文本
//         if (!decodedData.empty()) {
//             // 绘制二维码的边框
//             for (int i = 0; i < points.total(); i++) {
//                 line(frame, Point(points.at<Point2f>(i)[0], points.at<Point2f>(i)[1]), 
//                            Point(points.at<Point2f>((i + 1) % points.total())[0], points.at<Point2f>((i + 1) % points.total())[1]), 
//                            Scalar(0, 255, 0), 2);
//             }
//             // 在图像上绘制识别结果
//             putText(frame, "Decoded: " + decodedData, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
//         }

//         // 显示图像
//         imshow("Camera Feed", frame);

//         // 按下 'q' 键退出
//         if (waitKey(30) >= 0) break;
//     }

//     // 释放摄像头
//     cap.release();
//     destroyAllWindows();
//     return 0;
// }

#include <iostream>
#include <opencv2/opencv.hpp>
#include <zbar.h>

using namespace std;
using namespace cv;
using namespace zbar;

// 函数：识别二维码
string decodeQRCode(const Mat &image) {
    // 创建 ZBar 图像
    Image zbarImage(image.cols, image.rows, "Y800", image.data, image.cols * image.rows);

    // 创建 ZBar 扫描器
    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1); // 启用所有条形码类型

    // 扫描图像
    int n = scanner.scan(zbarImage);

    // 如果没有识别到二维码，返回空字符串
    if (n == 0) {
        return "";
    }

    // 获取识别结果
    for (Image::SymbolIterator symbol = zbarImage.symbol_begin(); symbol != zbarImage.symbol_end(); ++symbol) {
        return symbol->get_data(); // 返回识别到的二维码数据
    }

    return "";
}

int main() {
    // 打开摄像头
    VideoCapture cap(2); // 使用正确的摄像头索引
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera!" << endl;
        return 1;
    }

    while (true) {
        Mat frame;
        cap >> frame; // 从摄像头捕捉帧
        if (frame.empty()) {
            cerr << "Error: Could not grab a frame!" << endl;
            break;
        }

        // 将图像转换为灰度图
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // 解码二维码
        string decodedData = decodeQRCode(grayFrame);
        
        // 在帧上绘制识别结果
        if (!decodedData.empty()) {
            // 显示识别结果
            putText(frame, "Decoded: " + decodedData, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }

        // 显示图像
        imshow("Camera Feed", frame);

        // 按下 'q' 键退出
        if (waitKey(30) >= 0) break;
    }

    // 释放摄像头
    cap.release();
    destroyAllWindows();
    return 0;
}
