#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

// BRISK特征点检测
void briskKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int thresh = 30, int nOctaves = 4) {
    // 创建BRISK检测器
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(thresh, nOctaves);

    // 检测关键点和计算描述符
    brisk->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

//ORB特征点检测
void orbKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nfeatures = 500) {
    // 创建ORB检测器
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


//FLANN匹配特征点
enum FLANNMatcherType {
    FLANN_KDTREE,   // FLANN KD-Tree 用于 SIFT, SURF
    FLANN_LSH,      // FLANN LSH 用于 ORB, BRIEF, BRISK, AKAZE
    FLANN_LINEAR    // FLANN 线性扫描 任意类型
};
// 封装的通用 FLANN 匹配函数，支持交叉验证、多种匹配方法和描述子类型
std::vector<cv::DMatch> flannMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, FLANNMatcherType matcherType, bool useKnn = true, float ratioThresh = 0.75f, bool crossCheck = false) {
    std::vector<cv::DMatch> goodMatches;

    // 定义 FLANN 匹配器
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // 根据匹配类型选择合适的 FLANN 匹配器
switch (matcherType) {
    case FLANN_KDTREE: {
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::KDTreeIndexParams>(5));
        break;
    }
    case FLANN_LSH: {
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        break;
    }
    case FLANN_LINEAR: {
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LinearIndexParams>());
        break;
    }
    default:
        std::cerr << "Unsupported matcher type!" << std::endl;
        return goodMatches;
}
    // 正向匹配
    if (useKnn) {
        // 使用 KNN 进行匹配并应用 Ratio Test
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);  // K = 2

        // 进行 Ratio Test 过滤   --- 维度一致
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i].size() > 1 && knnMatches[i][1].distance > 0) {  // 确保有两个匹配并且第二个距离有效
                if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }
        }
    } else {
        // 使用普通的最近邻匹配
        matcher->match(descriptors1, descriptors2, goodMatches);
    }

    // // 交叉验证：反向匹配并过滤
    // if (crossCheck) {
    //     std::vector<cv::DMatch> reverseMatches;
    //     std::vector<cv::DMatch> crossCheckedMatches;

    //     // 反向匹配 (从 descriptors2 到 descriptors1)
    //     matcher->match(descriptors2, descriptors1, reverseMatches);

    //     // 保留正向匹配和反向匹配一致的匹配点
    //     for (size_t i = 0; i < goodMatches.size(); i++) {
    //         for (size_t j = 0; j < reverseMatches.size(); j++) {
    //             if (goodMatches[i].queryIdx == reverseMatches[j].trainIdx &&
    //                 goodMatches[i].trainIdx == reverseMatches[j].queryIdx) {
    //                 crossCheckedMatches.push_back(goodMatches[i]);
    //                 break;
    //             }
    //         }
    //     }

    //     return crossCheckedMatches;
    // }

    return goodMatches;
}


//RANSCAC匹配点优化，内点估计
class RANSACMatcher {
public:
    RANSACMatcher(double threshold, int maxIterations)
        : threshold(threshold), maxIterations(maxIterations) {}

    // 执行 RANSAC 匹配，计算内点
    std::vector<cv::DMatch> matchAndVisualize(const std::vector<cv::KeyPoint>& keypoints1,
                                              const std::vector<cv::KeyPoint>& keypoints2,
                                              const std::vector<cv::DMatch>& matches) {
        
        // 提取匹配的点
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // 运行 RANSAC 以找到内点
        std::vector<int> inliers = runRANSAC(points1, points2);

        // 输出内点数量
        std::cout << "Inliers: " << inliers.size() << std::endl;

        // 返回内点匹配
        return filterMatchesByInliers(matches, inliers);
    }

private:
    double threshold;       // 内点阈值
    int maxIterations;      // 最大迭代次数

    // 运行 RANSAC，使用 OpenCV 内置的 findHomography 方法
    std::vector<int> runRANSAC(const std::vector<cv::Point2f>& points1,
                               const std::vector<cv::Point2f>& points2) {
        std::vector<int> inliersMask;

        if (points1.size() < 4 || points2.size() < 4) {
            std::cerr << "Not enough points for RANSAC. At least 4 points are required." << std::endl;
            return inliersMask;
        }

        // 使用 OpenCV 内置 RANSAC 方法
        cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, threshold, inliersMask);

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

int main()
{
    cv::Mat img_1 = cv::imread("../images/origin/7.png",cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread("../images/origin/8.png",cv::IMREAD_COLOR);
    cv::Mat img_1_brisk, img_2_brisk;
    cv::Mat img_1_orb, img_2_orb;
    std::vector<cv::KeyPoint> brisk_keypoints_1, brisk_keypoints_2;
    std::vector<cv::KeyPoint> orb_keypoints_1, orb_keypoints_2;
    cv::Mat brisk_descriptors_1, brisk_descriptors_2;
    cv::Mat orb_descriptors_1, orb_descriptors_2;
    if (img_1.empty() || img_2.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
//BRISK特征点检测
    briskKeypointDetection(img_1, img_1_brisk, brisk_keypoints_1, brisk_descriptors_1, 30, 4);
    briskKeypointDetection(img_2, img_2_brisk, brisk_keypoints_2, brisk_descriptors_2, 30, 4);
std::cout << " BRISK keypoints in img1: " << brisk_descriptors_1.size() << std::endl;
std::cout << " BRISK keypoints in img2: " << brisk_descriptors_2.size() << std::endl;
//ORB特征点检测
    orbKeypointDetection(img_1, img_1_orb, orb_keypoints_1, orb_descriptors_1, 500);
    orbKeypointDetection(img_2, img_2_orb, orb_keypoints_2, orb_descriptors_2, 500);
std::cout << "Descriptors 1 size: " << orb_descriptors_1.size() << std::endl;
std::cout << "Descriptors 2 size: " << orb_descriptors_2.size() << std::endl;

// BRISK 特征点检测 + BFMatcher + HAMMING 距离匹配 + KNN 匹配特征点
    BFMatcherType brisk_BFmatcherType = BFMatcherType::HAMMING;
    BFMatchMethod brisk_BFmethod = BFMatchMethod::KNN;
    int brisk_k = 2;
    bool brisk_crossCheck = false;
    std::vector<cv::DMatch> brisk_matches = BFmatchFeatures(brisk_descriptors_1, brisk_descriptors_2, brisk_BFmatcherType, brisk_BFmethod, brisk_k, brisk_crossCheck);
    cv::Mat img_BF_BRISK_HAMMING_KNN;  // 修改后的图像命名
    cv::drawMatches(img_1, brisk_keypoints_1, img_2, brisk_keypoints_2, brisk_matches, img_BF_BRISK_HAMMING_KNN, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

// ORB 特征点检测 + BFMatcher + HAMMING2 距离匹配 + KNN 匹配特征点
    BFMatcherType orb_BFmatcherType = BFMatcherType::HAMMING2;
    BFMatchMethod orb_BFmethod = BFMatchMethod::KNN;
    int orb_k = 2;
    bool orb_crossCheck = false;
    std::vector<cv::DMatch> orb_matches = BFmatchFeatures(orb_descriptors_1, orb_descriptors_2, orb_BFmatcherType, orb_BFmethod, orb_k, orb_crossCheck);
    cv::Mat img_BF_ORB_HAMMING2_KNN;  // 修改后的图像命名
    cv::drawMatches(img_1, orb_keypoints_1, img_2, orb_keypoints_2, orb_matches, img_BF_ORB_HAMMING2_KNN, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

// BRISK 特征点检测 + FLANN + KNN 匹配特征点
    FLANNMatcherType brisk_flannMatcherType = FLANNMatcherType::FLANN_LSH;
    bool brisk_useKnn = true;
    float brisk_ratioThresh = 0.75f;
    bool brisk_crossCheck_flann = false;
    std::vector<cv::DMatch> brisk_flann_matches = flannMatch(brisk_descriptors_1, brisk_descriptors_2, brisk_flannMatcherType, brisk_useKnn, brisk_ratioThresh, brisk_crossCheck_flann);
    cv::Mat img_FLANN_BRISK_LSH;  // 修改后的图像命名
    cv::drawMatches(img_1, brisk_keypoints_1, img_2, brisk_keypoints_2, brisk_flann_matches, img_FLANN_BRISK_LSH, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

// ORB 特征点检测 + FLANN + KNN 匹配特征点
    FLANNMatcherType orb_flannMatcherType = FLANNMatcherType::FLANN_LSH;
    bool orb_useKnn = true;
    float orb_ratioThresh = 0.75f;
    bool orb_crossCheck_flann = false;
    std::vector<cv::DMatch> orb_flann_matches = flannMatch(orb_descriptors_1, orb_descriptors_2, orb_flannMatcherType, orb_useKnn, orb_ratioThresh, orb_crossCheck_flann);
    cv::Mat img_FLANN_ORB_LSH;  // 修改后的图像命名
    cv::drawMatches(img_1, orb_keypoints_1, img_2, orb_keypoints_2, orb_flann_matches, img_FLANN_ORB_LSH, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


//RANSAC匹配点优化，内点估计
    RANSACMatcher ransac(0.5, 1000);
    std::vector<cv::DMatch> brisk_ransac_matches;
    std::vector<cv::DMatch> orb_ransac_matches;
    brisk_ransac_matches = ransac.matchAndVisualize(brisk_keypoints_1, brisk_keypoints_2, brisk_flann_matches);
    orb_ransac_matches = ransac.matchAndVisualize(orb_keypoints_1, orb_keypoints_2, orb_flann_matches);
    cv::Mat img_RANSAC_BRISK_LSH;  // 修改后的图像命名
    cv::Mat img_RANSAC_ORB_LSH;  // 修改后的图像命名
    cv::drawMatches(img_1, brisk_keypoints_1, img_2, brisk_keypoints_2, brisk_ransac_matches, img_RANSAC_BRISK_LSH, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(img_1, orb_keypoints_1, img_2, orb_keypoints_2, orb_ransac_matches, img_RANSAC_ORB_LSH, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cv::namedWindow("img_1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_1_brisk", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_2_brisk", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_1_orb", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_2_orb", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_BF_BRISK_HAMMING_KNN", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_BF_ORB_HAMMING2_KNN", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_FLANN_BRISK_LSH", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_FLANN_ORB_LSH", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_RANSAC_BRISK_LSH", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("img_RANSAC_ORB_LSH", cv::WINDOW_AUTOSIZE);
    


    cv::resize(img_1, img_1, cv::Size(640, 480));
    cv::resize(img_2, img_2, cv::Size(640, 480));
    cv::resize(img_1_brisk, img_1_brisk, cv::Size(640, 480));
    cv::resize(img_2_brisk, img_2_brisk, cv::Size(640, 480));
    cv::resize(img_1_orb, img_1_orb, cv::Size(640, 480));
    cv::resize(img_2_orb, img_2_orb, cv::Size(640, 480));
    cv::resize(img_BF_BRISK_HAMMING_KNN, img_BF_BRISK_HAMMING_KNN, cv::Size(640, 480));
    cv::resize(img_BF_ORB_HAMMING2_KNN, img_BF_ORB_HAMMING2_KNN, cv::Size(640, 480));
    cv::resize(img_FLANN_BRISK_LSH, img_FLANN_BRISK_LSH, cv::Size(640, 480));
    cv::resize(img_FLANN_ORB_LSH, img_FLANN_ORB_LSH, cv::Size(640, 480));
    cv::resize(img_RANSAC_BRISK_LSH, img_RANSAC_BRISK_LSH, cv::Size(640, 480));
    cv::resize(img_RANSAC_ORB_LSH, img_RANSAC_ORB_LSH, cv::Size(640, 480));


    cv::imshow("img_1", img_1);
    cv::imshow("img_2", img_2);
    cv::imshow("img_1_brisk", img_1_brisk);
    cv::imshow("img_2_brisk", img_2_brisk);
    cv::imshow("img_1_orb", img_1_orb);
    cv::imshow("img_2_orb", img_2_orb);
    cv::imshow("img_BF_BRISK_HAMMING_KNN", img_BF_BRISK_HAMMING_KNN);
    cv::imshow("img_BF_ORB_HAMMING2_KNN", img_BF_ORB_HAMMING2_KNN);
    cv::imshow("img_FLANN_BRISK_LSH", img_FLANN_BRISK_LSH);
    cv::imshow("img_FLANN_ORB_LSH", img_FLANN_ORB_LSH);
    cv::imshow("img_RANSAC_BRISK_LSH", img_RANSAC_BRISK_LSH);   
    cv::imshow("img_RANSAC_ORB_LSH", img_RANSAC_ORB_LSH);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
