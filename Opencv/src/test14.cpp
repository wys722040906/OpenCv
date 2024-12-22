#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>

// 封装的物体跟踪函数
cv::Mat trackWithORB(const cv::Mat& prevImg, const cv::Mat& nextImg, cv::Rect& bbox, cv::Rect& bbox_sec, cv::Rect& bbox_origin)  {
    cv::Mat output = nextImg.clone();  // 输出图像

    // 1. 检测 ORB 特征点和计算描述子
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(prevImg, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(nextImg, cv::noArray(), keypoints2, descriptors2);

    // 2. 使用 BFMatcher 进行描述子的匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // 使用汉明距离
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 3. 过滤匹配：只保留前 N 个匹配
    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = static_cast<int>(matches.size() * 0.8); // 保留15%的好匹配
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // 4. 提取匹配的关键点
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // 5. 检查有效的匹配点数量
    if (points1.size() < 4 || points2.size() < 4) {
        std::cerr << "Not enough matching points to compute homography!" << std::endl;
        return output;  // 返回未处理的图像
    }

    // 6. 计算单应性矩阵并绘制结果
    cv::Mat inliers;
    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3, inliers);

    // 7. 绘制匹配结果
    for (const auto& match : matches) {
        cv::circle(output, keypoints2[match.trainIdx].pt, 5, cv::Scalar(0, 255, 0), -1); // 绘制匹配点
    }

    // 如果有足够的内点，绘制跟踪的矩形框
    if (!homography.empty()) {
        // 定义上一帧的矩形框四个角
        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(bbox_origin.x, bbox_origin.y); 
        corners[1] = cv::Point2f(bbox_origin.x + bbox_origin.width, bbox_origin.y);
        corners[2] = cv::Point2f(bbox_origin.x + bbox_origin.width, bbox_origin.y + bbox_origin.height);
        corners[3] = cv::Point2f(bbox_origin.x, bbox_origin.y + bbox_origin.height);

        // 每一帧都基于原始的 corners 进行透视变换
        std::vector<cv::Point2f> transformedCorners;
        cv::perspectiveTransform(corners, transformedCorners, homography);
        // bbox_sec = cv::boundingRect(transformedCorners); // 更新矩形框
        // 绘制当前帧的矩形框
        for (size_t i = 0; i < transformedCorners.size(); i++) {
            cv::line(output, transformedCorners[i], transformedCorners[(i + 1) % transformedCorners.size()], cv::Scalar(0, 0, 255), 2); // 绘制矩形框
        }
    }

    return output;
}

int main() {
    cv::VideoCapture cap(2);
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

    // 手动选择跟踪区域
    cv::Rect bbox = cv::selectROI("Select Object", frame1, false);
    cv::Rect bbox_sec;
    cv::Rect bbox_origin = bbox; // 记录原始的矩形框
    int loss_num = 0; // 丢失计数器
    if (bbox.width == 0 || bbox.height == 0) {
        std::cerr << "Invalid ROI selected, exiting." << std::endl;
        return 0;
    }

    // 创建跟踪器并初始化
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create(); // 选择跟踪器类型
    tracker->init(frame1, bbox); // 初始化跟踪器

    while (true) {
        cap >> frame2;
        if (frame2.empty()) {
            break;
        }

        // 使用 ORB 特征描述子的物体跟踪
        cv::Mat output = trackWithORB(frame1, frame2, bbox, bbox_sec, bbox_origin);

        // 更新跟踪器并绘制 ROI
        if (tracker->update(frame2, bbox)) {
            cv::rectangle(output, bbox, cv::Scalar(255, 0, 0), 2, 1); // 绘制跟踪框
        } else {
            // bbox.x = bbox_sec.x;
            // bbox.y = bbox_sec.y;
            // bbox.width = bbox_sec.width;
            // bbox.height = bbox_sec.height;
            // cv::rectangle(output, bbox, cv::Scalar(255, 0, 0), 2, 1); // 绘制跟踪框       
            std::cout << loss_num++ << "次" << "Tracking failed,statrt feature prediction!" << std::endl;
        }

        cv::imshow("Optical Flow with ORB", output);
        frame1 = frame2.clone(); // 更新 frame1
        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
