#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>

/*多类别SVM
One-vs-Rest (OvR) -- 多分类器，每个类别都有一个二分类器
One-vs-One (OvO) -- 多分类器，每个类别两两配对，共4个二分类器

每个类别分别准备正负样本数据

复杂场景：Yolo R-CNN

特征：
    1.图像分类
    颜色直方图：（如 RGB、HSV）颜色分布。
    纹理特征：GLCM（灰度共生矩阵）计算图像的纹理特征，例如对比度、熵等。
    边缘特征：Canny 边缘
    HOG（方向梯度直方图）特征：形状
    2. 文本分类
    TF-IDF（词频-逆文档频率）：衡量词在文档中的重要性。
    词嵌入：如 Word2Vec 或 GloVe 将单词映射到向量空间。
    N-grams：可以提取文本中连续的 n 个单词或字符。
    3. 时间序列分析
    移动平均：计算一段时间窗口内的平均值。
    时序特征：如日、周、月的时间戳特征。
    滞后变量：当前变量的前几期值。
    4. 数值特征
    原始特征：直接使用样本的数值属性，如房价预测中的面积、房间数等。
    归一化特征：通过 Min-Max 归一化或 Z-score 标准化处理后的特征。
*/

int main() {
    // 加载类别名称
    std::vector<std::string> classNames = {"class1", "class2", "class3"}; // 替换为实际类别

    // 存储模型路径
    std::vector<cv::Ptr<cv::ml::SVM>> models;

    // 加载每个类别的 SVM 模型
    for (const auto& className : classNames) {
        std::string modelPath = "path/to/your/models/" + className + "_svm_model.pkl";
        cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::load(modelPath);
        models.push_back(model);
    }

    // 读取输入图像
    cv::Mat inputImage = cv::imread("path/to/your/image.jpg", cv::IMREAD_GRAYSCALE); // 替换为实际图像路径
    if (inputImage.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // 调整图像大小，HOG通常使用64x128的输入图像
    cv::resize(inputImage, inputImage, cv::Size(64, 128));

    // 使用 HOGDescriptor 提取 HOG 特征
    cv::HOGDescriptor hog(
        cv::Size(64, 128), // 输入图像的大小
        cv::Size(16, 16),  // 滑动窗口大小
        cv::Size(8, 8),    // 滑动窗口步长
        cv::Size(8, 8),    // 单元格大小
        9);                // 梯度方向数

    std::vector<float> hogFeatures;
    hog.compute(inputImage, hogFeatures);

    // 将 HOG 特征转换为 cv::Mat 作为输入
    cv::Mat inputFeatures = cv::Mat(hogFeatures).reshape(1, 1); // 将一维特征变为行向量

    // 存储预测结果
    std::vector<float> predictions;
    for (const auto& model : models) {
        // 预测类别
        float prediction = model->predict(inputFeatures);
        predictions.push_back(prediction);
    }

    // 找到最大概率的类别
    auto maxIt = std::max_element(predictions.begin(), predictions.end());
    int predictedClassIndex = std::distance(predictions.begin(), maxIt);

    // 输出预测类别
    std::cout << "Predicted Class: " << classNames[predictedClassIndex] << std::endl;

    return 0;
}
