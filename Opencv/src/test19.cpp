#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;


//全局变量
// int lower_h = 61, lower_s = 94, lower_v = 87;
// int upper_h = 80, upper_s = 148, upper_v = 144;
int lower_h = 27, lower_s = 0, lower_v = 255;
int upper_h = 32, upper_s = 255, upper_v = 255;

int ErodeIter = 0;
int DialteIter = 0;
int OpenIter = 0;
int CloseIter = 0;
int CannyLowThresh = 0;
int CannyHighThresh = 0;

int white_balance_flag = 0;
int auto_wb_flag = 0;
int clahe_flag = 0;
int auto_clahe_flag = 0;

// 添加霍夫圆检测的全局参数
int hough_dp = 2;
int hough_minDist = 200;  // imgCanny.rows/8
int hough_param1 = 50;  // Canny边缘检测高阈值
int hough_param2 = 29;  // 累加器阈值
int hough_minRadius = 20;
int hough_maxRadius = 100;
int circle_area_thresh = 200;
int circle_ratio_thresh = 40;  // 实际值会除以100，所以0.4=40


void whiteBalance(cv::Mat image_input, cv::Mat& image) {
    image = image_input.clone();
    // 计算每个通道的平均值
    cv::Scalar mean = cv::mean(image); 
    double avgGray = (mean[0] + mean[1] + mean[2]) / 3;

    // 计算每个通道的增益
    double gainR = avgGray / mean[2]; // 红色通道的增益
    double gainG = 0.9*avgGray / mean[1]; // 绿色通道的增益
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



//HSV图二值化--处理图像--去掉背景细节--适合物体检测的
void img_pretreat(const Mat &src_, Mat &imgCanny, Mat &imgEdge, Scalar lower_hsv, Scalar upper_hsv,
                  Mat DialteKernel, Mat ErodeKernel, 
                  uint8_t ErodeIter, uint8_t DialteIter, uint8_t OpenIter, uint8_t CloseIter,
                  uint8_t CannyLowThresh, uint8_t CannyHighThresh){
    cv::Mat img_hsv, mask, src;

    src = src_.clone();
    //图像增强
    //白光补偿
    // cv::Mat white_img(src_.size(), src_.type());
    if (white_balance_flag) {
    whiteBalance(src_, src);  // 白平衡
    }
    //自适应白光补偿
    if (auto_wb_flag) {
    cv::Ptr<cv::xphoto::LearningBasedWB> wb = cv::xphoto::createLearningBasedWB();
    wb->setSaturationThreshold(0.8); // 饱和度阈值 0-1 越低：饱和像素越少，适于多彩  越高：饱和像素越多，影响更多饱和像素
    cv::Mat anvanced_wb_img(src_.size(), src_.type());
    wb->balanceWhite(src_, src);  // 白平衡   
    }

    //直方图均衡化
    if (clahe_flag) {
    std::vector<cv::Mat> channels;
    cv::split(src_, channels);
    cv::Ptr<cv::CLAHE> clahe_green = cv::createCLAHE();
    clahe_green->setTilesGridSize(cv::Size(2, 2)); // 块大小
    clahe_green->setClipLimit(2.0); // 限制量化范围  大：更多细节，对比度 小：更平滑  
    clahe_green->apply(channels[1], channels[1]);
    cv::merge(channels, src);
    }   
    //自适应直方图均衡化
    if (auto_clahe_flag) {
    std::vector<cv::Mat> channels;
    cv::split(src_, channels);
    cv::Ptr<cv::CLAHE> clahe_green = cv::createCLAHE();
    clahe_green->setTilesGridSize(cv::Size(2, 2)); // 块大小
    clahe_green->setClipLimit(2.0); // 限制量化范围  大：更多细节，对比度 小：更平滑  
    clahe_green->apply(channels[1], channels[1]);
    cv::merge(channels, src);
    }

    // imshow("src", src);


    cv::cvtColor(src, img_hsv, cv::COLOR_BGR2HSV);

    cv::inRange(img_hsv, lower_hsv, upper_hsv, mask);

    //GaussianBlur
    // GaussianBlur(mask, mask, Size(3, 3), 0);
    // 保存原始的Canny边缘检测结果
    // cv::Canny(mask, imgEdge, CannyLowThresh, CannyHighThresh);
    // 对mask进行形态学处理
    // mask = imgEdge.clone();
    //形态学操作
    /*
    膨胀（Dilation）：可以扩大前景物体的面积，用于连接断裂的边缘或填充小洞。
    腐蚀（Erosion）：可以缩小前景物体的面积，用于去除小噪声点。
    开运算（Opening）：先腐蚀后膨胀，用于去除小噪声区域。
    闭运算（Closing）：先膨胀后腐蚀，用于填充小孔洞和连接分离的区域。  
    */
    for(uint i = 0; i < DialteIter; i++){
        cv::dilate(mask, mask, DialteKernel);
    }
    for(uint i = 0; i < ErodeIter; i++){
        cv::erode(mask, mask, ErodeKernel);
    }

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    for(uint i = 0; i < OpenIter; i++){
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element); //开运算--去噪点
    }
    for(uint i = 0; i < CloseIter; i++){
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element); //闭运算--连通分离
    }
        //Canny边缘检测
    
    imgCanny = mask;
    // waitKey(10);
}


void find_contours(const Mat &imgCanny, vector<vector<Point>> &contours, vector<Vec4i> &hierarchy, int mode, int method,
                  const Scalar &RecColor, const int RecAreaThread) {
    cv::findContours(imgCanny, contours, hierarchy, mode, method); //mode:轮廓检索模式：cv::RETR_EXTERNAL：只检测最外层的轮廓。cv::RETR_TREE：检测所有轮廓，并建立轮廓之间的等级关系。 ，method:轮廓近似办法:cv::CHAIN_APPROX_NONE：保留所有的轮廓点。cv::CHAIN_APPROX_SIMPLE：压缩水平、垂直和对角线，留存轮廓的端点。

    //轮廓处理
    for (int i = 0; i < contours.size(); i++) {

    //形状过滤--圆形
        // double aspect_ratio = boundingRect(contours[i]).width / boundingRect(contours[i]).height;
        // if (cv::contourArea(contours[i]) > RecAreaThread && fabs(aspect_ratio - 1.0) < 0.2) {
        //     // 仅保留接近正圆的物体
        // }
    // 面积过滤
        if (cv::contourArea(contours[i]) > RecAreaThread) { //轮廓面积大于阈值
            // cv::RotatedRect minRect = cv::minAreaRect(contours[i]); //最小外接矩形
            // cv::Point2f rect_points[4];
            // minRect.points(rect_points); //获取最小外接矩形的四个顶点
            // cv::rectangle(imgCanny, rect_points[0], rect_points[2], RecColor, 2, 8, 0); //绘制最小外接矩形
            // cv::drawContours(imgCanny, contours, i, RecColor, 5, 8, hierarchy, 0); //绘制轮廓
            continue;
        }
        else {
            contours.erase(contours.begin() + i); //删除面积小于阈值的轮廓
            i--;    
        }
    }

}


int main()
{
    VideoCapture cap(2);
    // VideoCapture cap("/home/wys/Desktop/Project/VisionProject/Opencv/images/1.mp4");
    if (!cap.isOpened()){
        cout << "Cannot open camera" << endl;
        return -1;
    }
    namedWindow("Live Video", WINDOW_AUTOSIZE);
    namedWindow("Canny", WINDOW_AUTOSIZE);


    // 创建滑动条
    // createTrackbar("Lower H", "Live Video", &lower_h, 180);
    // createTrackbar("Lower S", "Live Video", &lower_s, 255);
    // createTrackbar("Lower V", "Live Video", &lower_v, 255);
    // createTrackbar("Upper H", "Live Video", &upper_h, 180);
    // createTrackbar("Upper S", "Live Video", &upper_s, 255);
    // createTrackbar("Upper V", "Live Video", &upper_v, 255);

    // createTrackbar("Erode Iter", "Live Video", &ErodeIter, 10);
    // createTrackbar("Dialte Iter", "Live Video", &DialteIter, 10);
    // createTrackbar("Open Iter", "Live Video", &OpenIter, 10);
    // createTrackbar("Close Iter", "Live Video", &CloseIter, 10);
    // createTrackbar("Canny Low Thresh", "Live Video", &CannyLowThresh, 255);
    // createTrackbar("Canny High Thresh", "Live Video", &CannyHighThresh, 255);

    // createTrackbar("White Balance", "Live Video", &white_balance_flag, 1);
    // createTrackbar("Auto White Balance", "Live Video", &auto_wb_flag, 1);
    // createTrackbar("CLAHE", "Live Video", &clahe_flag, 1);
    // createTrackbar("Auto CLAHE", "Live Video", &auto_clahe_flag, 1);

    // 在main函数中添加新的trackbar
    createTrackbar("Hough DP", "Live Video", &hough_dp, 5);
    createTrackbar("Hough MinDist", "Live Video", &hough_minDist, 200);
    createTrackbar("Hough Param1", "Live Video", &hough_param1, 300);
    createTrackbar("Hough Param2", "Live Video", &hough_param2, 200);
    createTrackbar("Hough MinRadius", "Live Video", &hough_minRadius, 100);
    createTrackbar("Hough MaxRadius", "Live Video", &hough_maxRadius, 200);
    createTrackbar("Circle Area Thresh", "Live Video", &circle_area_thresh, 200);
    createTrackbar("Circle Ratio Thresh", "Live Video", &circle_ratio_thresh, 100);

    cv::Mat imgCanny = cv::Mat::zeros(Size(640, 480), CV_8UC1);
    cv::Mat imgEdge = cv::Mat::zeros(Size(640, 480), CV_8UC1);  // 新增
    cv::Mat DialteKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat ErodeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // uint8_t ErodeIter = 0;
    // uint8_t DialteIter = 0;
    // uint8_t OpenIter = 1;
    // uint8_t CloseIter = 1;
    // uint8_t CannyLowThresh = 50;
    // uint8_t CannyHighThresh = 150;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int mode = cv::RETR_EXTERNAL;
    int method = cv::CHAIN_APPROX_SIMPLE;
    Scalar RecColor(0, 255, 0);
    int RecAreaThread = 400;  //轮廓面积阈值


    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()){
            cout << "Error: empty frame" << endl;
            break;
        }
        resize(frame, frame, Size(640, 480));

        cv::Scalar lower_hsv(lower_h, lower_s, lower_v);  //代替了循环调用回调 | 进程调用回调 -- 变量在全局被访问
        cv::Scalar upper_hsv(upper_h, upper_s, upper_v);

        img_pretreat(frame, imgCanny, imgEdge, lower_hsv, upper_hsv, 
                DialteKernel, ErodeKernel, ErodeIter, DialteIter, 
                OpenIter, CloseIter, CannyLowThresh, CannyHighThresh);

        // //霍夫圆检测
        vector<Vec3f> circles;
        HoughCircles(imgCanny, circles, HOUGH_GRADIENT, 
                    std::max(1, hough_dp),  // dp不能为0
                    std::max(1, hough_minDist),  // minDist不能为0
                    hough_param1, 
                    hough_param2,
                    hough_minRadius, 
                    hough_maxRadius);
        // 查找轮廓（用于计算圆度）
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(imgCanny.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 绘制符合条件的圆
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            
            // 创建圆形mask来获取当前圆的轮廓
            Mat circleMask = Mat::zeros(imgCanny.size(), CV_8UC1);
            circle(circleMask, center, radius, Scalar(255), -1);
            
            // 找到与当前圆重叠最多的轮廓
            int bestContourIdx = -1;
            double maxOverlap = 0;
            
            for (size_t j = 0; j < contours.size(); j++) {
                Rect boundRect = boundingRect(contours[j]);
                if (center.x >= boundRect.x && center.x <= boundRect.x + boundRect.width &&
                    center.y >= boundRect.y && center.y <= boundRect.y + boundRect.height) {
                    
                    double area = contourArea(contours[j]);
                    if (area > circle_area_thresh) {
                        double aspect_ratio = (double)boundRect.width / boundRect.height;
                        if (fabs(aspect_ratio - 1.0) < circle_ratio_thresh/100.0) {
                            bestContourIdx = j;
                            break;
                        }
                    }
                }
            }
            
            // 如果找到合适的轮廓，绘制圆
            if (bestContourIdx >= 0) {
                // 绘制圆心
                circle(frame, center, 3, Scalar(0, 255, 0), -1);
                // 绘制圆轮廓
                circle(frame, center, radius, Scalar(0, 255, 0), 2);
            }
        }

        find_contours(imgCanny, contours, hierarchy, mode, method, RecColor, RecAreaThread);    
        
        // for(auto i = 0; i < contours.size(); i++){
        //     // cv::drawContours(frame, contours, i, RecColor, 5, 8, hierarchy, 0); //绘制轮廓
        //     cv::RotatedRect minRect = cv::minAreaRect(contours[i]); //最小外接矩形
        //     cv::Point2f rect_points[4];
        //     minRect.points(rect_points); //获取最小外接矩形的四个顶点
        //     cv::rectangle(frame, rect_points[0], rect_points[2], RecColor, 2, 8, 0); //绘制最小外接矩形
        // }

        imshow("Live Video", frame);
        imshow("Canny", imgCanny);
        imshow("Edge", imgEdge);  // 添加这行来显示原始边缘

        if (waitKey(1) == 20)
        {
            break;
        }
    }
    return 0;
}
