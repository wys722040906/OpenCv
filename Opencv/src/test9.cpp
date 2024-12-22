#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/video/tracking.hpp>

// 定义高级光流方法的类型
enum OpticalFlowMethod {
    LUCAS_KANADE,  // Lucas-Kanade 稀疏光流
    FARNEBACK,     // Farneback 稠密光流
    DUAL_TVL1,      // Dual TV-L1 稠密光流
    TWO_WAY        // 双向光流
};

// 封装的光流跟踪函数
cv::Mat trackOpticalFlow(const cv::Mat& prevImg, const cv::Mat& nextImg, OpticalFlowMethod method) {
    cv::Mat output;

    // 1. 确保输入图像为灰度图像
    cv::Mat grayPrevImg, grayNextImg;
    if (prevImg.channels() == 3) {
        cv::cvtColor(prevImg, grayPrevImg, cv::COLOR_BGR2GRAY);
    } else {
        grayPrevImg = prevImg;
    }

    if (nextImg.channels() == 3) {
        cv::cvtColor(nextImg, grayNextImg, cv::COLOR_BGR2GRAY);
    } else {
        grayNextImg = nextImg;
    }

    // 2. 确保图像大小一致
    if (grayPrevImg.size() != grayNextImg.size()) {
        std::cerr << "Error: The size of the two frames does not match!" << std::endl;
        return cv::Mat();  // 返回空矩阵以表示错误
    }

    switch (method) {
        case LUCAS_KANADE: {
            // 使用 Lucas-Kanade 光流法
            std::vector<cv::Point2f> prevPts, nextPts, backPts;
            std::vector<uchar> status, status_back;
            std::vector<float> err, err_back;

            // 3. 使用 Shi-Tomasi 角点检测找到前一帧中的特征点
            cv::goodFeaturesToTrack(grayPrevImg, prevPts, 100, 0.01, 10);

            // 4. 计算正向光流（从 prevImg 到 nextImg）
            cv::calcOpticalFlowPyrLK(grayPrevImg, grayNextImg, prevPts, nextPts, status, err);

            // 5. 计算反向光流（从 nextImg 到 prevImg）
            cv::calcOpticalFlowPyrLK(grayNextImg, grayPrevImg, nextPts, backPts, status_back, err_back);

            // 6. 双向光流一致性检查：过滤那些误差较大的点
            std::vector<cv::Point2f> goodPrevPts, goodNextPts;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && status_back[i]) {
                    // 检查正向和反向光流的距离是否小于某个阈值
                    float dist = cv::norm(prevPts[i] - backPts[i]);
                    if (dist < 1.0) {  // 设定阈值为1像素，可以根据需要调整
                        goodPrevPts.push_back(prevPts[i]);
                        goodNextPts.push_back(nextPts[i]);
                    }
                }
            }

            // 7. 绘制结果
            cv::cvtColor(grayPrevImg, output, cv::COLOR_GRAY2BGR);
            for (size_t i = 0; i < goodPrevPts.size(); i++) {
                cv::circle(output, goodPrevPts[i], 3, cv::Scalar(0, 255, 0), -1);  // 绘制特征点
                cv::line(output, goodPrevPts[i], goodNextPts[i], cv::Scalar(0, 0, 255), 2);  // 绘制光流线
            }
            break;
        }

        case FARNEBACK: {
            // 使用 Farneback 稠密光流法
            cv::Mat flow;//两通道二维场量
            cv::calcOpticalFlowFarneback(grayPrevImg, grayNextImg, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            // 将稠密光流结果转为彩色显示
            // 幅值:饱和度--鲜艳程度，越大越鲜艳
            // 角度:色调--不同颜色
            cv::Mat flowParts[2], magnitude, angle;
            cv::split(flow, flowParts);  // 分割 x 和 y 分量
            cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);  //极坐标系 计算角度和幅值

            // 调试输出大小信息
            std::cout << "Magnitude size: " << magnitude.size() << std::endl;
            std::cout << "Angle size: " << angle.size() << std::endl;

            // 将角度和幅值转换为 8 位无符号类型
            cv::Mat magnitude_8U, angle_8U;
            magnitude.convertTo(magnitude_8U, CV_8UC1, 255.0 / cv::norm(magnitude, cv::NORM_INF));  // 归一化并转换
            angle.convertTo(angle_8U, CV_8UC1, 255.0 / 360.0);  // 角度范围 [0, 360] 转换为 [0, 255]

            // 确保尺寸一致
            if (magnitude_8U.size() != angle_8U.size()) {
                std::cerr << "Error: Magnitude and angle sizes do not match!" << std::endl;
                break;
            }

            cv::Mat hsvSplit[3], hsvImg, bgrImg;
            hsvSplit[0] = angle_8U;  // Hue 表示光流方向
            hsvSplit[1] = magnitude_8U;  // Saturation 表示光流大小
            hsvSplit[2] = cv::Mat::ones(angle.size(), CV_8UC1) * 255;  // Value

            // 检查尺寸
            std::cout << "HSV Hue size: " << hsvSplit[0].size() << std::endl;
            std::cout << "HSV Saturation size: " << hsvSplit[1].size() << std::endl;

            // 合并 HSV 通道
            cv::merge(hsvSplit, 3, hsvImg);  // 合并为 HSV 图像
            cv::cvtColor(hsvImg, bgrImg, cv::COLOR_HSV2BGR);  // 转为 BGR 格式

            output = bgrImg;  // 结果为稠密光流的彩色图像
            break;
        }


        case DUAL_TVL1: {
            // 使用 Dual TV-L1 稠密光流法（未启用，作为占位符）
            // 你可以在这里实现 Dual TV-L1 算法，如果你的 OpenCV 构建支持 optflow 模块
            break;
        }

        case TWO_WAY: {
            // 双向光流法，先计算正向光流，再计算反向光流，然后取平均
            cv::Mat forwardFlow = trackOpticalFlow(prevImg, nextImg, FARNEBACK);
            cv::Mat backwardFlow = trackOpticalFlow(nextImg, prevImg, FARNEBACK);

            if (!forwardFlow.empty() && !backwardFlow.empty()) {
                cv::addWeighted(forwardFlow, 0.5, backwardFlow, 0.5, 0, output);
            }
            break;
        }

        default:
            std::cerr << "Unknown optical flow method!" << std::endl;
            break;
    }

    return output;
}


int main()   
{   
    cv::VideoCapture cap(2);
    cap.set(cv::CAP_PROP_FORMAT, CV_8UC1);
    if (!cap.isOpened()) {      
        std::cerr << "Cannot open camera!" << std::endl;      
        return -1;   
    }   

    cv::Mat frame1, frame2;   
    cap >> frame1;   
    if (frame1.empty()) {
        std::cerr << "Error: Could not read frame from camera!" << std::endl;
        return -1;
    }   

    while (true) {   
        cap >> frame2;   
        if (frame2.empty()) {   
            break;   
        }   
        // cv::imshow("Frame1", frame1);
        cv::imshow("Frame2", frame2);  
        cv::Mat flow = trackOpticalFlow(frame1, frame2, LUCAS_KANADE);   
        cv::imshow("Optical Flow", flow);   

        frame1 = frame2.clone(); // 使用 clone 来确保 frame1 保持最新的图像数据
        if (cv::waitKey(10) == 27) {   
            break;   
        }   
    }   

    cap.release();   
    cv::destroyAllWindows();   
    return 0;   
}  
