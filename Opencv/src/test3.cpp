#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
cv::Mat adaptiveBilateralFilter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat dst = src.clone();
    cv::Mat temp;

    // Convert the source image to grayscale if it is not already
    if (src.channels() == 3) {
        cv::cvtColor(src, temp, cv::COLOR_BGR2GRAY);
    } else {
        temp = src.clone();
    }

    // Calculate the variance of the image
    cv::Mat mean, stddev;
    cv::meanStdDev(temp, mean, stddev);

    // Adjust the sigma values based on the image variance
    double var = stddev.at<double>(0) * stddev.at<double>(0);
    double adaptiveSigmaColor = sigmaColor * var;
    double adaptiveSigmaSpace = sigmaSpace * var;

    // Apply bilateral filter
    cv::bilateralFilter(src, dst, d, adaptiveSigmaColor, adaptiveSigmaSpace);

    return dst;
}
int main() {
    cv::Mat image = cv::imread("/home/wys/Vision Project/Opencv/1.png");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }
    cv::namedWindow("mean_filter",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("gaussia_filter",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("median_filter",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("two_edge_filter",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("sobelX",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("sobelY",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("ScharrX",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("ScharrY",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("laplacian",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("canny",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("auto_two_edge_filter",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("fastNLMeans",cv::WINDOW_AUTOSIZE);

    //均值
    cv::Mat mean_filter;
    int kernel = 5;
    cv::blur(image, mean_filter, cv::Size(kernel,kernel)); 
    //高斯
    cv::Mat gaussia_filter;
    cv::GaussianBlur(image,  gaussia_filter, cv::Size(kernel,kernel),0);
    //中值
    cv::Mat median_filter;
    cv::medianBlur(image, median_filter, 5);
    //双边滤波
    cv::Mat two_edge_filter;
    cv::bilateralFilter(image, two_edge_filter, 9, 75, 75);
    //Sobel梯度计算
    cv::Mat sobelX, sobelY;
    cv::Sobel(image, sobelX, CV_64F, 1, 0, 5);
    cv::Sobel(image, sobelY, CV_64F, 0, 1, 5);
    //Scharr算子
    cv::Mat ScharrX;
    cv::Mat ScharrY;
    cv::Scharr(image, ScharrX, CV_64F, 1, 0);
    cv::Scharr(image, ScharrY, CV_64F, 0, 1);
    //Laplaciance算子
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_64F);
    //Cany算法
    cv::Mat canny;
    cv::Canny(image, canny, 100, 200);
    //自适应双边滤波
    cv::Mat auto_two_edge_filter;
    auto_two_edge_filter = adaptiveBilateralFilter(image, 9, 75, 75);
    //(滤波后)自适应阈值处理--更好二值化
    // cv::adaptiveThreshold()
    //非局部均值去噪
    cv::Mat fastNLMeans;
    cv::fastNlMeansDenoisingColored(image, fastNLMeans, 10, 10, 7, 21);

    cv::resize(mean_filter,mean_filter, cv::Size(640, 480));
    cv::resize(gaussia_filter,gaussia_filter, cv::Size(640, 480));
    cv::resize(median_filter,median_filter, cv::Size(640, 480));
    cv::resize(two_edge_filter,two_edge_filter, cv::Size(640, 480));
    cv::resize(sobelX,sobelX, cv::Size(640, 480));
    cv::resize(sobelY,sobelY, cv::Size(640, 480));
    cv::resize(ScharrX,ScharrX, cv::Size(640, 480));
    cv::resize(ScharrY,ScharrY, cv::Size(640, 480));
    cv::resize(laplacian,laplacian, cv::Size(640, 480));
    cv::resize(canny,canny, cv::Size(640, 480));
    cv::resize(auto_two_edge_filter,auto_two_edge_filter, cv::Size(640, 480));
    cv::resize(fastNLMeans,fastNLMeans, cv::Size(640, 480));
    cv::resize(image, image, cv::Size(640, 480));

    cv::imshow("mean_filter", mean_filter);
    cv::imshow("gaussia_filter",gaussia_filter );
    cv::imshow("median_filter",median_filter );
    cv::imshow("two_edge_filter",two_edge_filter );
    cv::imshow("sobelX",sobelX );
    cv::imshow("sobelY",sobelY);
    cv::imshow("ScharrX",ScharrX );
    cv::imshow("ScharrY",ScharrY );
    cv::imshow("laplacian",laplacian );
    cv::imshow("canny",canny );
    cv::imshow("auto_two_edge_filter",auto_two_edge_filter );
    cv::imshow("fastNLMeans",fastNLMeans );
    cv::imshow("image", image);

    cv::waitKey(0);
    return 0;
}
