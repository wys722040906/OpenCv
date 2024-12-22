#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>


// 提取颜色直方图作为全局特征
cv::Mat extractColorHistogram(const cv::Mat& img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);  // 转换为 HSV 色彩空间

    // 计算直方图
    int h_bins = 50, s_bins = 60;  // H 和 S 分量的 bins 数
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());  // L1 归一化

    return hist.reshape(1, 1);  // 将直方图展开为一维向量
}

// 提取 ORB 特征描述子
cv::Mat extractORBDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat descriptors;
    // 创建ORB检测器
    // 使用带金字塔参数的ORB
    // int nfeatures = 500;  // 最大特征数
    // float scaleFactor = 1.2f;  // 金字塔尺度因子
    // int nlevels = 8;  // 金字塔层数
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    return descriptors;
}

// 使用 KMeans 聚类生成视觉词汇表（BoVW）
cv::Mat buildVocabulary(const std::vector<cv::Mat>& descriptors_list, int dictionarySize) {
    // 将所有描述子垂直堆叠
    cv::Mat descriptors_all;
    for (const auto& descriptors : descriptors_list) {
        if (!descriptors.empty()) {
            descriptors_all.push_back(descriptors);
        }
    }

    // 设置 KMeans 聚类参数
    cv::TermCriteria tc(cv::TermCriteria::MAX_ITER, 100, 0.001);
    int retries = 10;  // 尝试多次，选择最佳的聚类结果
    int flags = cv::KMEANS_RANDOM_CENTERS;

    // 执行 KMeans 聚类
    cv::Mat vocabulary;
    cv::kmeans(descriptors_all, dictionarySize, cv::noArray(), tc, retries, flags, vocabulary);

    return vocabulary;
}

// 计算图像的 Bag-of-Visual-Words 特征
cv::Mat computeBOWDescriptor(const cv::Mat& descriptors, const cv::Mat& vocabulary) {
    cv::Mat bowDescriptor = cv::Mat::zeros(1, vocabulary.rows, CV_32F);

    // 对每个描述子，找到最近的视觉词
    for (int i = 0; i < descriptors.rows; i++) {
        cv::Mat descriptor = descriptors.row(i);

        // 计算描述子与所有视觉词的距离
        cv::Mat distances;
        for (int j = 0; j < vocabulary.rows; j++) {
            double dist = cv::norm(descriptor, vocabulary.row(j), cv::NORM_HAMMING);
            distances.push_back(dist);
        }

        // 找到最小距离的索引
        double minVal;
        int minIdx;
        cv::minMaxIdx(distances, &minVal, 0, &minIdx);

        // 增加对应视觉词的计数
        bowDescriptor.at<float>(0, minIdx) += 1;
    }

    // 归一化直方图
    cv::normalize(bowDescriptor, bowDescriptor, 1, 0, cv::NORM_L1);

    return bowDescriptor;
}

// 多特征融合并进行匹配
void matchUsingCombinedFeatures(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& vocabulary) {
    // 提取颜色直方图
    cv::Mat hist1 = extractColorHistogram(img1);
    cv::Mat hist2 = extractColorHistogram(img2);

    // 提取 ORB 描述子
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1 = extractORBDescriptors(img1, keypoints1);
    cv::Mat descriptors2 = extractORBDescriptors(img2, keypoints2);

    // 计算 BoW 描述子
    cv::Mat bowDescriptor1 = computeBOWDescriptor(descriptors1, vocabulary);
    cv::Mat bowDescriptor2 = computeBOWDescriptor(descriptors2, vocabulary);

    // 特征融合（特征连接）
    cv::Mat combinedFeature1, combinedFeature2;
    cv::hconcat(hist1, bowDescriptor1, combinedFeature1);
    cv::hconcat(hist2, bowDescriptor2, combinedFeature2);

    // 计算融合特征之间的距离（欧氏距离）
    double distance = cv::norm(combinedFeature1 - combinedFeature2, cv::NORM_L2);
    std::cout << "Combined Feature Distance: " << distance << std::endl;

    // 根据距离判断是否匹配（阈值需要根据数据集调整）
    double threshold = 0.5;  // 示例阈值
    if (distance < threshold) {
        std::cout << "Images are considered matching based on fused features." << std::endl;
    } else {
        std::cout << "Images are not matching based on fused features." << std::endl;
    }
}


int main() {
    // 读取图像数据集并提取 ORB 描述子（用于构建词汇表）
    std::vector<cv::Mat> images;  // 假设您有一个图像数据集
    std::vector<cv::Mat> descriptors_list;

    // 示例：加载两个图像（您可以扩展为多个图像）
    cv::Mat img1 = cv::imread("image1.jpg");
    cv::Mat img2 = cv::imread("image2.jpg");

    if (img1.empty() || img2.empty()) {
        std::cerr << "Cannot load images!" << std::endl;
        return -1;
    }

    images.push_back(img1);
    images.push_back(img2);

    // 提取所有图像的 ORB 描述子
    for (const auto& img : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors = extractORBDescriptors(img, keypoints);
        descriptors_list.push_back(descriptors);
    }

    // 构建视觉词汇表 视觉词汇的大小 过小：泛化 过大：过拟合 简单 50-100 复杂 1000
    int dictionarySize = 100;  
    cv::Mat vocabulary = buildVocabulary(descriptors_list, dictionarySize);

    // 使用融合特征进行匹配
    matchUsingCombinedFeatures(img1, img2, vocabulary);

    return 0;
}