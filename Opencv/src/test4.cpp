#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <opencv2/xphoto.hpp>


void gammaCorrection(const cv::Mat& input, cv::Mat& output, double gamma) {
    // 创建查找表（LUT）
    uchar lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
    }

    // 应用查找表
    output.create(input.size(), input.type());
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            output.at<uchar>(y, x) = lut[input.at<uchar>(y, x)];
        }
    }
}
void whiteBalance(cv::Mat image_input, cv::Mat& image) {
    image = image_input.clone();
    // 计算每个通道的平均值
    cv::Scalar mean = cv::mean(image); 
    double avgGray = (mean[0] + mean[1] + mean[2]) / 3;

    // 计算每个通道的增益
    double gainR = avgGray / mean[2]; // 红色通道的增益
    double gainG = avgGray / mean[1]; // 绿色通道的增益
    double gainB = avgGray / mean[0]; // 蓝色通道的增益

    // 应用增益
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);
            pixel[0] = cv::saturate_cast<uchar>(pixel[0] * gainB); // B
            pixel[1] = cv::saturate_cast<uchar>(pixel[1] * gainG); // G
            pixel[2] = cv::saturate_cast<uchar>(pixel[2] * gainR); // R
        }
    }
}

int main() {
    cv::Mat image = cv::imread("/home/wys/Desktop/Project/Vision Project/Opencv/images/噪声图.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_color = cv::imread("/home/wys/Desktop/Project/Vision Project/Opencv/images/噪声图.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }


// 计算直方图 --- 图像展现
    // int histSize = 256;
    // float range[] = {0, 256};
    // const float* histRange = {range};
    // cv::Mat hist;
    // cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    // // 归一化直方图
    // cv::Mat histImage(histSize, histSize, CV_8UC1, cv::Scalar(0));
    // cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);
    // // 绘制直方图
    // for (int i = 1; i < histSize; ++i) {
    //     cv::line(histImage, cv::Point(i - 1, histSize - cvRound(hist.at<float>(i - 1))),
    //              cv::Point(i, histSize - cvRound(hist.at<float>(i))),
    //              cv::Scalar(255), 2, 8, 0);
    // }
    // cv::imshow("Histogram", histImage);

//直方图均衡化 --- 图像增强  
    cv::Mat equallist_img ;
    cv::equalizeHist(image, equallist_img);
//某通道直方图均衡化 --- 图像增强
    std::vector<cv::Mat> channels;
    cv::split(image_color, channels);
    cv::equalizeHist(channels[1], channels[1]);
    cv::merge(channels, image_color);
//自适应直方图均衡化 --- 图像增强
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setTilesGridSize(cv::Size(2, 2)); // 块大小
    clahe->setClipLimit(2.0); // 限制量化范围  大：更多细节，对比度 小：更平滑  
    cv::Mat autoequallist_img;
    clahe->apply(image, autoequallist_img);
//某通道·自适应直方图均衡化 --- 图像增强
    std::vector<cv::Mat> channels;
    cv::split(image_color, channels);
    cv::Ptr<cv::CLAHE> clahe_green = cv::createCLAHE();
    clahe_green->setTilesGridSize(cv::Size(2, 2)); // 块大小
    clahe_green->setClipLimit(2.0); // 限制量化范围  大：更多细节，对比度 小：更平滑  
    clahe_green->apply(channels[1], channels[1]);
    cv::merge(channels, image_color);
//伽马校正 --- 图像增强---对比度增强
    cv::Mat gamma_img;
    gammaCorrection(image, gamma_img, 0.5); // 伽马值 1.5 效果最好  大：提高亮度 小：突出细节
//普通白平衡 --- 图像增强---色调增强
    cv::Mat white_img(image_color.size(), image_color.type());
    whiteBalance(image_color, white_img);  // 白平衡
//学习型白平衡 --- 图像增强---色调增强
    cv::Ptr<cv::xphoto::LearningBasedWB> wb = cv::xphoto::createLearningBasedWB();
    wb->setSaturationThreshold(0.8); // 饱和度阈值 0-1 越低：饱和像素越少，适于多彩  越高：饱和像素越多，影响更多饱和像素
    cv::Mat anvanced_wb_img(image_color.size(), image_color.type());
    wb->balanceWhite(image_color, anvanced_wb_img);  // 白平衡
//转换到HSV空间 --- 饱和度(2) 明度(3)增强
    cv::Mat hsv_img;
    cv::cvtColor(image_color, hsv_img, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_img, hsv_channels);
    hsv_channels[2] = hsv_channels[2] * 1.2; // 明度增强
    hsv_channels[1] = hsv_channels[1] * 1.5; // 饱和度增强
    cv::threshold(hsv_channels[2], hsv_channels[2], 255, 255, cv::THRESH_TRUNC); // 明度阈值
    cv::threshold(hsv_channels[1], hsv_channels[1], 255, 255, cv::THRESH_TRUNC); // 饱和度阈值
    cv::merge(hsv_channels, hsv_img);
    cv::cvtColor(hsv_img, image_color, cv::COLOR_HSV2RGB); // 转换回BGR空间
//Cany边缘检测 --- 叠加
    cv::Mat canny_img;
    cv::Canny(image, canny_img, 50, 150); // lowThreshold, highThreshold -- 边缘检测灵敏度
    cv::Mat canny_color_img;
    cv::cvtColor(canny_img, canny_color_img, cv::COLOR_GRAY2BGR);
    cv::addWeighted(image_color, 0.8, canny_color_img, 0.2, 0, canny_color_img); // 伽马 : 叠加到图象上的常量


    cv::namedWindow("origin", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("color_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("histImage", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("autoequallist_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("gamma_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("white_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("anvanced_wb_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("hsv_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("canny_color_img", cv::WINDOW_AUTOSIZE);


    cv::resize( image, image, cv::Size(640, 480));
    cv::resize( image_color, image_color, cv::Size(640, 480));
    cv::resize( equallist_img, equallist_img, cv::Size(640, 480));
    cv::resize( autoequallist_img, autoequallist_img, cv::Size(640, 480));
    cv::resize( gamma_img, gamma_img, cv::Size(640, 480));
    cv::resize( white_img, white_img, cv::Size(640, 480));
    cv::resize( anvanced_wb_img, anvanced_wb_img, cv::Size(640, 480));
    cv::resize( hsv_img, hsv_img, cv::Size(640, 480));
    cv::resize( canny_color_img, canny_color_img, cv::Size(640, 480));
 

    cv::imshow("origin", image);    
    cv::imshow("color_img", image_color);
    cv::imshow("histImage", equallist_img);
    cv::imshow("autoequallist_img", autoequallist_img);
    cv::imshow("gamma_img", gamma_img);
    cv::imshow("white_img", white_img);
    cv::imshow("anvanced_wb_img", anvanced_wb_img);
    cv::imshow("hsv_img", hsv_img);
    cv::imshow("canny_color_img", canny_color_img);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
