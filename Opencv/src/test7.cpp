#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

void orbKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nfeatures = 500) {
    // 创建ORB检测器
    // 使用带金字塔参数的ORB
    // int nfeatures = 500;  // 最大特征数
    // float scaleFactor = 1.2f;  // 金字塔尺度因子
    // int nlevels = 8;  // 金字塔层数
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);

    // 检测关键点和计算描述符
    orb->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

//BFMatcher匹配特征点
// 匹配类型的枚举
enum class BFMatcherType {
    L1,
    L2,
    HAMMING,
    HAMMING2
};
// 匹配方法的枚举
enum class BFMatchMethod {
    NEAREST_NEIGHBOR,  // 最近邻匹配
    KNN,               // K-近邻匹配
    CROSS_CHECK        // 交叉验证匹配
};
// 封装的BFMatcher匹配函数
std::vector<cv::DMatch> BFmatchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                                      BFMatcherType matcherType, BFMatchMethod method,
                                      int k = 2, bool crossCheck = false) {
    // 选择匹配器类型
    cv::Ptr<cv::BFMatcher> matcher;
    switch (matcherType) {
        case BFMatcherType::L1:
            matcher = cv::BFMatcher::create(cv::NORM_L1, crossCheck);
            break;
        case BFMatcherType::L2:
            matcher = cv::BFMatcher::create(cv::NORM_L2, crossCheck);
            break;
        case BFMatcherType::HAMMING:
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, crossCheck);
            break;
        case BFMatcherType::HAMMING2:
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING2, crossCheck);
            break;
        default:
            throw std::invalid_argument("Unsupported matcher type");
    }

    std::vector<cv::DMatch> matches;

    // 匹配方法
    if (method == BFMatchMethod::NEAREST_NEIGHBOR) {
        // 使用最近邻匹配
        matcher->match(descriptors1, descriptors2, matches);
    } else if (method == BFMatchMethod::KNN) {
        // 使用 KNN 匹配
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptors1, descriptors2, knnMatches, k);

        // Lowe's ratio test 用于过滤匹配结果
        const float ratio_thresh = 0.75f;  // 设置 ratio test 阈值
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                matches.push_back(knnMatches[i][0]);  // 保留通过 ratio test 的匹配
            }
        }
    } else if (method == BFMatchMethod::CROSS_CHECK) {
        // 启用交叉验证匹配
        matcher->match(descriptors1, descriptors2, matches);
    } else {
        throw std::invalid_argument("Unsupported match method");
    }

    return matches;
}


//RANSCAC匹配点优化，内点估计
class RANSACMatcher {
public:
    RANSACMatcher(double threshold, int maxIterations)
        : threshold(threshold), maxIterations(maxIterations) {}

    // 执行 RANSAC 匹配，计算内点
    std::vector<cv::DMatch> matchAndVisualize(const std::vector<cv::KeyPoint>& keypoints1,
                                              const std::vector<cv::KeyPoint>& keypoints2,
                                              const std::vector<cv::DMatch>& matches,
                                              cv::Mat& H) {
        
        // 提取匹配的点
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // 运行 RANSAC 以找到内点
        std::vector<int> inliers = runRANSAC(points1, points2, H);

        // 输出内点y掩码(匹配子)数量
        std::cout << "inliera_mask size: " << inliers.size() << std::endl;

        // 返回内点匹配
        if(matches.size())
        return filterMatchesByInliers(matches, inliers);
        else return std::vector<cv::DMatch>();
    }

private:
    double threshold;       // 内点阈值
    int maxIterations;      // 最大迭代次数

    // 运行 RANSAC，使用 OpenCV 内置的 findHomography 方法
    std::vector<int> runRANSAC(const std::vector<cv::Point2f>& points1,
                               const std::vector<cv::Point2f>& points2,
                               cv::Mat& H) {
        std::vector<int> inliersMask = std::vector<int>(points1.size(), 0);

        if (points1.size() < 4 || points2.size() < 4) {
            std::cerr << "Not enough points for RANSAC. At least 4 points are required." << std::endl;
            return inliersMask;
        }

        // 使用 OpenCV 内置 RANSAC 方法
        H = cv::findHomography(points1, points2, cv::RANSAC, threshold, inliersMask);

        if (H.empty()) {
            std::cerr << "Homography calculation failed." << std::endl;
            return inliersMask;
        }

        // 返回内点的掩码
        return inliersMask;
    }

    // 根据内点掩码返回内点的匹配对
    std::vector<cv::DMatch> filterMatchesByInliers(const std::vector<cv::DMatch>& matches,
                                                   const std::vector<int>& inliersMask) {
        std::vector<cv::DMatch> goodMatches;
        for (size_t i = 0; i < matches.size(); ++i) {
            if (inliersMask[i]) {
                goodMatches.push_back(matches[i]);
            }
        }
        return goodMatches;
    }
};



int main() {

    cv::VideoCapture cap("/home/wys/Desktop/Project/VisionProject/Opencv/images/GreebRabbish.mp4");
    if(!cap.isOpened()) {
        std::cout << "Can not open video file." << std::endl;
        return -1;
    }
    cv::Mat frame;
    while(cap.read(frame)) {
    if(frame.empty())    break;
    cv::Mat obj_img = cv::imread("../images/origin/1.png",cv::IMREAD_COLOR);
    // cv::Mat scene_img = cv::imread("../images/origin/8.png",cv::IMREAD_COLOR);
    cv::Mat scene_img = frame.clone();

    cv::Mat obj_output, scene_output;
    if(obj_img.empty() || scene_img.empty()) {
        std::cout << "Can not open image file." << std::endl;        
        return -1;
    }
    std::vector<cv::KeyPoint> obj_keypoints, scene_keypoints;
    cv::Mat obj_descriptors, scene_descriptors;
    orbKeypointDetection(obj_img, obj_output, obj_keypoints, obj_descriptors);
    orbKeypointDetection(scene_img, scene_output, scene_keypoints, scene_descriptors);

    // 匹配特征点
    // BFMatcherType matcherType = BFMatcherType::HAMMING;
    // BFMatchMethod method = BFMatchMethod::KNN;
    // int k = 2;
    // bool crossCheck = false;
    std::vector<cv::DMatch> good_matches = BFmatchFeatures(obj_descriptors, scene_descriptors, BFMatcherType::HAMMING, BFMatchMethod::KNN, 2, false);

    //单应性矩阵
    RANSACMatcher matcher(0.5, 1000);
    cv::Mat H;
    std::vector<cv::DMatch> inliers_matches = matcher.matchAndVisualize(obj_keypoints, scene_keypoints, good_matches, H);

    std::vector<cv::Point2f> obj_points = std::vector<cv::Point2f>();
    std::vector<cv::Point2f> scene_points = std::vector<cv::Point2f>();


    if(good_matches.size() >= 4 && H.empty() == false) {
        for(const auto& match : good_matches) {
        obj_points.push_back(obj_keypoints[match.queryIdx].pt);
        scene_points.push_back(scene_keypoints[match.trainIdx].pt);
        cv::circle(obj_img, obj_keypoints[match.queryIdx].pt, 10, cv::Scalar(0, 0 , 255), -1);
        cv::circle(scene_img, scene_keypoints[match.trainIdx].pt, 10, cv::Scalar(0, 0 , 255), -1);
        }
        cv::Rect bounding_rect = cv::boundingRect(obj_points);
        std::vector<cv::Point2f> obj_points_rect(4), scene_points_rect(4);
        obj_points_rect[0] = cv::Point2f(bounding_rect.x, bounding_rect.y);
        obj_points_rect[1] = cv::Point2f(bounding_rect.x + bounding_rect.width, bounding_rect.y);
        obj_points_rect[2] = cv::Point2f(bounding_rect.x + bounding_rect.width, bounding_rect.y + bounding_rect.height);
        obj_points_rect[3] = cv::Point2f(bounding_rect.x, bounding_rect.y + bounding_rect.height);
        cv::perspectiveTransform(obj_points_rect, scene_points_rect, H);
        //转换为整数
        std::vector<cv::Point> obj_points_int, scene_points_int;
        for (const auto& pt : obj_points) {
            obj_points_int.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }
        for (const auto& pt : scene_points) {
            scene_points_int.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }
        // 绘制物体检测结果
        cv::line(obj_img, obj_points_rect[0], obj_points_rect[1], cv::Scalar(0, 255, 0), 4);
        cv::line(obj_img, obj_points_rect[1], obj_points_rect[2], cv::Scalar(0, 255, 0), 4);
        cv::line(obj_img, obj_points_rect[2], obj_points_rect[3], cv::Scalar(0, 255, 0), 4);
        cv::line(obj_img, obj_points_rect[3], obj_points_rect[0], cv::Scalar(0, 255, 0), 4);
        cv::line(scene_img, scene_points_rect[0], scene_points_rect[1], cv::Scalar(0, 255, 0), 4);
        cv::line(scene_img, scene_points_rect[1], scene_points_rect[2], cv::Scalar(0, 255, 0), 4);
        cv::line(scene_img, scene_points_rect[2], scene_points_rect[3], cv::Scalar(0, 255, 0), 4);
        cv::line(scene_img, scene_points_rect[3], scene_points_rect[0], cv::Scalar(0, 255, 0), 4);
        // cv::polylines(obj_img, std::vector<std::vector<cv::Point>>{obj_points_int}, true, cv::Scalar(0, 255, 0), 2);
        // cv::polylines(scene_img, std::vector<std::vector<cv::Point>>{scene_points_int}, true, cv::Scalar(0, 255, 0), 2);
    }else std::cout << "Not enough matches are found - %d/%d" << good_matches.size() << std::endl;
        




    cv::namedWindow("obj_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("scene_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("obj_output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("scene_output", cv::WINDOW_AUTOSIZE);
    
    cv::resize(obj_img, obj_img, cv::Size(640, 480));
    cv::resize(scene_img, scene_img, cv::Size(640, 480));
    cv::resize(frame, frame, cv::Size(640, 480));
    cv::resize(obj_output, obj_output, cv::Size(640, 480));
    cv::resize(scene_output, scene_output, cv::Size(640, 480));
    
    cv::imshow("obj_img", obj_img);
    cv::imshow("scene_img", scene_img);
    cv::imshow("video", frame);
    cv::imshow("obj_output", obj_output);
    cv::imshow("scene_output", scene_output);

    cv::waitKey(50);
    }

    return 0;
}