# 范数

# 单应性矩阵

**图像上某点到某点的投影矩阵，RANSAC得出** -- **至少四点**

```
cv::Mat H = cv::findHomography(std<vector::Point2f>points1, points2, cv::RANSAC);
```

应用：

- 图像配准：一个图像对齐到另一个图像上，特别是在视角变化、平移、旋转等情况下

```
cv::warpPerspective(img1, result, H, img2.size());
```

- 全景图拼接

数学原理
$$
\begin{aligned}x' &= \frac{h_{11} x + h_{12} y + h_{13}}{h_{31} x + h_{32} y + h_{33}} \\
y '&= \frac{h_{21} x + h_{22} y + h_{23}}{h_{31} x + h_{32} y + h_{33}}
\end{aligned}
$$


# 基础矩阵

- 3×3 的矩阵，它描述了两个图像平面之间的本质关系，描述了两个摄像机视角之间的几何关系

- **本质矩阵**：一对立体图像中的点对 (x1,x2)(x_1, x_2)(x1,x2)，这个矩阵捕捉了这些点如何在两个视角中对应起来
- **图像坐标**：x1,y1)和 x2,y2 是在第一张图像和第二张图像中的点的齐次坐标。齐次坐标通常表示为 (x,y,1)T，这里的  1是齐次坐标的最后一个分量，用于将二维坐标扩展到三维空1间。


$$
x_2^T F x_1 = 0 \\
x_1 = \begin{pmatrix}x_1 \\y_1 \\1\end{pmatrix} \\x_2 = \begin{pmatrix}x_2 \\y_2 \\1\end{pmatrix} \\
\text{Error} = x_2^T F x_1
$$

**极几何约束**，它说明了两个图像中对应的点对之间的几何关系：

- **x1x_1x1** 和 **x2x_2x2** 是在两个图像中的对应点的齐次坐标。
- **FFF** 是本质矩阵。
- 这个方程描述了点 x1x_1x1 和 x2x_2x2 如何通过本质矩阵 FFF 相互约束。

```
// 计算本质矩阵
cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
// 验证方程 x2^T F x1 = 0 是否成立
for (size_t i = 0; i < points1.size(); ++i) {
cv::Mat x1 = (cv::Mat_<double>(3,1) << points1[i].x, points1[i].y, 1.0);
cv::Mat x2 = (cv::Mat_<double>(3,1) << points2[i].x, points2[i].y, 1.0);
double error = cv::Mat(x2.t() * F * x1).at<double>(0, 0);
std::cout << "Error for point pair " << i << ": " << error << std::endl;
}
```

## 应用

**立体视觉**：基础矩阵用于立体图像对的校准和深度计算。

**图像拼接**：在图像拼接和全景图生成中，基础矩阵用于图像对齐。

**相机标定**：基础矩阵是相机标定过程中的一个重要步骤。

## 比较

**单应性矩阵** H 是一个 3×3矩阵，用于描述两个图像平面之间的透视变换关系，相机位置固定或仅发生平移、旋转、缩放时使用

- 需要知道相机的内参和外参来计算准确的变换矩阵。
- 适用于平面图像间的透视变换，能够描述图像间的平移、旋转、缩放等。
- 用于描述图像间的透视变换，适用于相机内参已知的情况。

**基础矩阵 **F 3×3 矩阵，描述两个图像之间的本质几何关系。它基于相机的内部和外部参数，适用于处理不同视角下的图像对

- 不需要知道相机的内参（焦距、主点位置等），仅使用匹配的点对。
- 用于描述图像间的本质几何关系，适用于相机内参未知的情况。
- 秩为 2，并且具有 7 个自由度（一个矩阵的 9 个参数，去掉一个，因为矩阵的行或列是线性相关的）。

# 描述符

## **定义**:

- 描述符是一个特征点的特征向量，用于编码特征点周围区域的图像信息。它通常包含多个维度，每个维度表示图像区域的某种特征。
- 描述符是特征点的数值表示，允许在不同图像中匹配相同的特征点。

## **数据结构**:

- 在 OpenCV 中，`descriptors` 通常是一个 `cv::Mat` 类型的矩阵。每一行对应一个特征点的描述符，每一列对应描述符的一个维度。
- 描述符矩阵的维度取决于特征检测算法。不同算法生成的描述符维度不同。例如：
- - **SIFT**: 生成 128 维的描述符。
  - **SURF**: 生成 64 维或 128 维的描述符。
  - **ORB**: 生成 32 维的描述符。

## 二进制描述子 vs 普通描述子

#### 特点：

- **数据类型**: 使用浮点数表示描述符，通常为 `float` 类型。
- **大小**: 普通描述子通常较大，因为它们用浮点数表示每个维度的特征。例如，SIFT 描述符有 128 维，每维为 32 位浮点数，总大小为 128 字节。
- **匹配速度**: 普通描述子的匹配速度通常较慢，因为它们使用欧几里得距离或其他度量方式进行比较，这需要计算浮点数之间的差异。
- **准确性**: 浮点描述子通常能提供更高的匹配精度，因为它们包含更多的特征信息和细节。

| 特征           | 二进制描述子                     | 普通描述子                             |
| -------------- | -------------------------------- | -------------------------------------- |
| **数据类型**   | 二进制（0 和 1 或 0x00 和 0xFF） | 浮点数（float）                        |
| **描述符大小** | 较小（如 256 位，32 字节）       | 较大（如 128 维，128 字节）            |
| **匹配速度**   | 较快（使用 Hamming 距离）        | 较慢（使用欧几里得距离或其他度量方式） |
| **存储效率**   | 较高（占用内存少）               | 较低（占用内存多）                     |
| **匹配精度**   | 较低（信息损失较多）             | 较高（包含更多特征信息）               |

### 使用场景

- **二进制描述子**: 适用于对速度要求较高的实时应用，如实时图像处理和机器人导航等。由于其计算和存储效率较高，适合在资源有限的环境中使用。
- **普通描述子**: 适用于对匹配精度要求较高的应用，如图像识别和检索等。虽然计算和存储开销较大，但能够提供更高的匹配精度和鲁棒性。

# 对比度

- 图像中亮度值（灰度值）的差异程度，对比度衡量了图像中不同区域的明暗差异。
- 高对比度的图像通常具有明显的明暗差异，而低对比度的图像则可能显得较为平淡和灰暗。

## 计算

$$
\text{Contrast} = \frac{\text{Max\_Intensity} - \text{Min\_Intensity}}{\text{Max\_Intensity} + \text{Min\_Intensity}}
$$

## 作用

- 虑低对比度点，减噪，保留稳定特征-->提高质量
- 高对比度点(边缘 角点)，不同角度，尺度更为稳定

# 图像金字塔

## 构建

**高斯模糊**：对图像进行高斯模糊，生成平滑的图像。

**下采样**：将图像缩小一半，生成分辨率更低的图像。

**重复步骤 1 和 2**：继续对缩小后的图像进行高斯模糊和下采样，生成更低分辨率的图像。

注：原始图像会生成多个不同分辨率的图像，这些图像组成一个金字塔结构。每个分辨率层称为一个 `octave`，在每个 `octave` 中，可以进一步生成不同平滑程度的图像，这些图像称为 `scale`

## 特征描述

在特征检测算法（如 SIFT、SURF 等）中，关键点可以在不同的尺度上检测到。每个关键点的 `octave` 属性表示它是在图像金字塔的哪个层级（即哪个分辨率）上检测到的。

- **较高层级的关键点**：表示在较低分辨率的图像上检测到的关键点。这些关键点通常对应于图像中较大的结构或特征。
- **较低层级的关键点**：表示在较高分辨率的图像上检测到的关键点。这些关键点通常对应于图像中较小的细节或特征

## 八度尺寸

**`nOctaveLayers`**：

- **定义**：每个 octave 内部的尺度层数。每个 octave 包含多个尺度的图像，即在同一分辨率层级下，对图像进行不同程度的模糊处理。
- **默认值**：4。这意味着每个 octave 内部进行 4 次不同的模糊处理，以捕捉不同尺度的特征。
- **作用**：决定了每个 octave 中的特征检测精度。更多的尺度层可以帮助更精确地检测特征，但也会增加计算量。

## 实现

```
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat img = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 检测关键点
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img, keypoints);

    // 打印一些关键点的信息
    for (const auto& kp : keypoints) {
        std::cout << "KeyPoint: ["
                  << "pt: (" << kp.pt.x << ", " << kp.pt.y << "), "
                  << "size: " << kp.size << ", "
                  << "angle: " << kp.angle << ", "
                  << "response: " << kp.response << ", "
                  << "octave: " << kp.octave << ", "
                  << "class_id: " << kp.class_id << "]" << std::endl;
    }

    return 0;
}

```



# 欧几里得距离

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**定义**: 计算两个描述符向量之间的直线距离。

**公式**:  其中 x 和 y是两个描述符向量，n 是描述符的维度。

**应用**: 常用于 SIFT 和 SURF 描述符，因为这些描述符是**基于浮点数的特征向量**，适合用欧几里得距离来计算相似性。

# 余弦相似度

$$
\text{cosine\_similarity}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
$$

**定义**: 测量两个向量之间的夹角来计算相似度，常用于高维稀疏特征描述符。

**公式**: 其中 x⋅y是向量 x 和 y 的点积，∥x∥ 和 ∥y∥ 是向量的模。

**应用**: 通常用于描述符是**稀疏的二进制特征向量**，如 ORB 和 BRIEF 描述符。

# 汉明距离

$$
d(x, y) = \text{Number of differing bits between } x \text{ and } y
$$

**定义**: 衡量两个二进制描述符之间的不同位的数量。

**公式**: 

**应用**: **专用于二进制描述符**，如 BRIEF 和 ORB 描述符，适用于 Hamming 距离来计算相似性。

# 曼哈顿距离

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

**定义**: 计算描述符向量在各个维度上的绝对差值之和。

**公式**: 

**应用**: 适用于基于浮点数的特征描述符，特别是当特征向量不是归一化的情况下。

# 卡方距离

$$
d(x, y) = \frac{1}{2} \sum_{i=1}^{n} \frac{(x_i - y_i)^2}{x_i + y_i}
$$

- **定义**: 衡量两个描述符的统计差异。
- **公式**: 
- **应用**: 主要用于描述符的特征值是**直方图或频率分布**时，例如使用 BRIEF 描述符时。

# 离散差分算法

- 计算离散信号或图像中导数的一种方法--计算图像的梯度,以检测图像中的边缘.
- Sobel 算子、Prewitt 算子和 Roberts

- 基本原理：通过计算邻近像素值之间的差异来估算图像的导数

一维
$$
f'[i] = f[i+1] - f[i]
$$
二维-X
$$
G_x = I * \begin{bmatrix} -1 & 1 \end{bmatrix}
$$
二维-Y
$$
G_y = I * \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

# 卷积核

也称为**滤波器**或**核矩阵**）是一个小的矩阵，用于图像处理中的卷积运算

**选择卷积核**：定义一个小的矩阵，例如 3×33 \times 33×3 或 5×55 \times 55×5 的矩阵。

**卷积核定位**：将卷积核的中心与图像的某个像素对齐。

**计算点积**：将卷积核的每个元素与图像中相应位置的像素值相乘，然后将这些乘积求和，得到一个新的像素值。

**移动卷积核**：将卷积核移动到图像中的下一个位置，重复步骤 3。

**输出结果**：将计算得到的新像素值放入输出图像的相应位置。

```
cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
// 定义输出图像矩阵
cv::Mat result;
// 应用卷积
cv::filter2D(image, result, CV_64F, kernel);
```

# 梯度

- 梯度--图像中灰度值变化的度量，能够指示图像在某一方向上的变化率
- 边缘检测、特征提取和图像增强等任务

# 非极大值抑制

## 梯度幅值

- 衡量图像中每个像素位置的**梯度强度**的一个度量，本质上是图像灰度值的**变化率**，通常用于边缘检测，因为图像中的边缘对应着灰度值变化较大的区域
- 计算

 Sobel 算子-->在 x 和 y 方向上计算图像灰度值的导数。--> x 和 y 方向的梯度图Gx,Gy
$$
G = \sqrt{G_x^2 + G_y^2}
$$

$$
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

- 非极大值抑制的核心原理

假设有一个像素点 (x,y)(x, y)(x,y) 及其梯度方向 θ\thetaθ，可以根据梯度方向将相邻的像素划分为四种情况：

1. θ 约为 0 度：与水平方向相邻的两个像素比较。
2. θ约为 45 度：与对角方向相邻的两个像素比较。
3. θ 约为 90 度：与垂直方向相邻的两个像素比较。
4. θ 约为 135 度：与另一个对角方向相邻的两个像素比较。

## 非极大值抑制实现

**cv::cartToPolar(grad_x, grad_y, gradientMagnitude, gradientDirection, true);**

```
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

void nonMaxSuppression(cv::Mat& gradientMagnitude, cv::Mat& gradientDirection, cv::Mat& output) {
    output = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);

    for (int i = 1; i < gradientMagnitude.rows - 1; ++i) {
        for (int j = 1; j < gradientMagnitude.cols - 1; ++j) {
            float direction = gradientDirection.at<float>(i, j);
            float magnitude = gradientMagnitude.at<float>(i, j);
            
            float magnitude1, magnitude2;

            if ((direction >= -22.5 && direction <= 22.5) || (direction >= 157.5 || direction <= -157.5)) {
                magnitude1 = gradientMagnitude.at<float>(i, j - 1);
                magnitude2 = gradientMagnitude.at<float>(i, j + 1);
            } else if ((direction > 22.5 && direction < 67.5) || (direction < -112.5 && direction > -157.5)) {
                magnitude1 = gradientMagnitude.at<float>(i - 1, j + 1);
                magnitude2 = gradientMagnitude.at<float>(i + 1, j - 1);
            } else if ((direction >= 67.5 && direction <= 112.5) || (direction >= -112.5 && direction <= -67.5)) {
                magnitude1 = gradientMagnitude.at<float>(i - 1, j);
                magnitude2 = gradientMagnitude.at<float>(i + 1, j);
            } else {
                magnitude1 = gradientMagnitude.at<float>(i - 1, j - 1);
                magnitude2 = gradientMagnitude.at<float>(i + 1, j + 1);
            }

            if (magnitude >= magnitude1 && magnitude >= magnitude2) {
                output.at<uchar>(i, j) = static_cast<uchar>(magnitude);
            }
        }
    }
}

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 高斯滤波
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 1.4);

    // 计算梯度
    cv::Mat grad_x, grad_y;
    cv::Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, grad_y, CV_32F, 0, 1, 3);

    // 计算梯度幅值和方向
    cv::Mat gradientMagnitude, gradientDirection;
    cv::cartToPolar(grad_x, grad_y, gradientMagnitude, gradientDirection, true);

    // 非极大值抑制
    cv::Mat nonMaxSuppressed;
    nonMaxSuppression(gradientMagnitude, gradientDirection, nonMaxSuppressed);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Non-Max Suppressed", nonMaxSuppressed);

    // 等待按键
    cv::waitKey(0);
    return 0;
}

```

## 梯度幅值实现

**void cv::convertScaleAbs(InputArray src, OutputArray dst, double alpha = 1, double beta = 0);**

- `src`：输入数组，可以是多通道数组。
- `dst`：输出数组，和输入数组具有相同的大小和类型。
- `alpha`：可选的缩放因子，默认值为 1。
- `beta`：可选的加数，默认值为 0。

绝对值转换为 8 位无符号整数，并进行缩放和偏移

```C++
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat computeGradientMagnitude(const cv::Mat& src) {
    cv::Mat gray, grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;

    // Convert the image to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Compute the gradient in the x direction
    cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // Compute the gradient in the y direction
    cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Compute the gradient magnitude
    cv::Mat grad_x_squared, grad_y_squared;
    cv::pow(grad_x, 2, grad_x_squared);
    cv::pow(grad_y, 2, grad_y_squared);
    cv::sqrt(grad_x_squared + grad_y_squared, grad);

    return grad;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cout << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Compute gradient magnitude
    cv::Mat grad = computeGradientMagnitude(src);

    // Display the result
    cv::imshow("Original Image", src);
    cv::imshow("Gradient Magnitude", grad);
    cv::waitKey(0);

    return 0;
}
```



# 双阈值检测+边缘追踪

- 细分边缘：

  区分强边缘、弱边缘和非边缘--通过两个不同的阈值将边缘像素分为强边缘、弱边缘和抑制掉的像素

**设定高低阈值**：设定两个阈值，一个是高阈值 ，另一个是低阈值 

**强边缘**：任何梯度幅值大于高阈值的像素被认为是强边缘。

**弱边缘**：任何梯度幅值在低阈值和高阈值之间的像素被认为是弱边缘。

**非边缘**：任何梯度幅值低于低阈值的像素被抑制（设为零）。

**边缘连接**：连接弱边缘和强边缘。只有与强边缘连接的弱边缘被保留为真正的边缘。

**初始化**：从强边缘开始，通过队列或递归的方式遍历连接的弱边缘。

**遍历弱边缘**：对于每一个强边缘像素，检查其 8 邻域像素。如果邻域中有弱边缘像素，则将其标记为强边缘，并继续检查其邻域。

**完成边缘连接**：遍历所有强边缘像素后，剩余的弱边缘像素将被抑制为非边缘。

**实现**

```
#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>

// 边缘跟踪函数
void edgeTrackingByHysteresis(cv::Mat& edges, cv::Mat& weakEdges, int weakVal, int strongVal) {
    std::queue<cv::Point> edgePoints;

    // 将所有强边缘像素点加入队列
    for (int i = 1; i < edges.rows - 1; ++i) {
        for (int j = 1; j < edges.cols - 1; ++j) {
            if (edges.at<uchar>(i, j) == strongVal) {
                edgePoints.push(cv::Point(j, i));
            }
        }
    }

    // 处理队列中的每一个强边缘像素点
    while (!edgePoints.empty()) {
        cv::Point p = edgePoints.front();
        edgePoints.pop();

        // 检查 8 邻域中的弱边缘像素
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int y = p.y + i;
                int x = p.x + j;

                if (weakEdges.at<uchar>(y, x) == weakVal) {
                    // 将弱边缘标记为强边缘
                    edges.at<uchar>(y, x) = strongVal;
                    // 将这个点加入队列
                    edgePoints.push(cv::Point(x, y));
                }
            }
        }
    }

    // 抑制所有未连接的弱边缘像素
    for (int i = 0; i < edges.rows; ++i) {
        for (int j = 0; j < edges.cols; ++j) {
            if (edges.at<uchar>(i, j) != strongVal) {
                edges.at<uchar>(i, j) = 0;
            }
        }
    }
}

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 高斯滤波
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 1.4);

    // 计算梯度
    cv::Mat grad_x, grad_y;
    cv::Sobel(blurred, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(blurred, grad_y, CV_16S, 0, 1, 3);

    // 计算梯度幅值和方向
    cv::Mat gradientMagnitude, gradientDirection;
    cv::cartToPolar(grad_x, grad_y, gradientMagnitude, gradientDirection, true);

    // 转换为 CV_8U 类型
    cv::Mat gradMagnitude8U;
    gradientMagnitude.convertTo(gradMagnitude8U, CV_8U);

    // 非极大值抑制
    cv::Mat nonMaxSuppressed = cv::Mat::zeros(gradMagnitude8U.size(), CV_8U);
    for (int i = 1; i < gradMagnitude8U.rows - 1; ++i) {
        for (int j = 1; j < gradMagnitude8U.cols - 1; ++j) {
            float direction = gradientDirection.at<float>(i, j);
            float magnitude = gradMagnitude8U.at<uchar>(i, j);

            float magnitude1, magnitude2;

            if ((direction >= -22.5 && direction <= 22.5) || (direction >= 157.5 || direction <= -157.5)) {
                magnitude1 = gradMagnitude8U.at<uchar>(i, j - 1);
                magnitude2 = gradMagnitude8U.at<uchar>(i, j + 1);
            } else if ((direction > 22.5 && direction < 67.5) || (direction < -112.5 && direction > -157.5)) {
                magnitude1 = gradMagnitude8U.at<uchar>(i - 1, j + 1);
                magnitude2 = gradMagnitude8U.at<uchar>(i + 1, j - 1);
            } else if ((direction >= 67.5 && direction <= 112.5) || (direction >= -112.5 && direction <= -67.5)) {
                magnitude1 = gradMagnitude8U.at<uchar>(i - 1, j);
                magnitude2 = gradMagnitude8U.at<uchar>(i + 1, j);
            } else {
                magnitude1 = gradMagnitude8U.at<uchar>(i - 1, j - 1);
                magnitude2 = gradMagnitude8U.at<uchar>(i + 1, j + 1);
            }

            if (magnitude >= magnitude1 && magnitude >= magnitude2) {
                nonMaxSuppressed.at<uchar>(i, j) = magnitude;
            }
        }
    }

    // 双阈值检测
    double lowThreshold = 50;
    double highThreshold = 150;
    cv::Mat edges = cv::Mat::zeros(nonMaxSuppressed.size(), CV_8U);
    cv::Mat weakEdges = cv::Mat::zeros(nonMaxSuppressed.size(), CV_8U);

    for (int i = 0; i < nonMaxSuppressed.rows; ++i) {
        for (int j = 0; j < nonMaxSuppressed.cols; ++j) {
            uchar pixel = nonMaxSuppressed.at<uchar>(i, j);
            if (pixel >= highThreshold) {
                edges.at<uchar>(i, j) = 255;  // 强边缘
            } else if (pixel >= lowThreshold) {
                weakEdges.at<uchar>(i, j) = 255;  // 弱边缘
            }
        }
    }

    // 边缘跟踪
    edgeTrackingByHysteresis(edges, weakEdges, 255, 255);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Edges", edges);

    // 等待按键
    cv::waitKey(0);
    return 0;
}

```



# 噪声

## 高斯噪声

- 白噪声，符合正态分布

**std::random_device rd;**

真随机数生成器，生成种子对象，用于其他伪随机数生成器生成种子

**std::mt19937 gen(rd());**    

用rd()返回随机数作为种子初始化伪随机数生成器--每次运行时伪随机数序列不同

**std::normal_distribution<> d(mean, stddev);**

模板类的高斯分布--均值，标注差

**noise.at< cv::Vec3b >(i, j)[0] = cv::saturate_cast<  uchar>(d(gen));**

gen-->随机x,d()-->随机y,cv::saturate_cast--归一化

**cv::addWeighted(src, 1.0, noise, 1.0, 0.0, dst);**

`src`: 第一个输入图像。

`1.0`: 第一个图像的权重。

`noise`: 第二个输入图像（噪声图像）。

`1.0`: 第二个图像的权重。

`0.0`: 标量值，添加到加权图像的和上。

`dst`: 输出图像。

```
#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

// Function to add Gaussian noise to an image
cv::Mat addGaussianNoise(const cv::Mat &src, double mean, double stddev) {
    cv::Mat noise(src.size(), src.type());
    cv::Mat dst;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);

    for (int i = 0; i < noise.rows; ++i) {
        for (int j = 0; j < noise.cols; ++j) {
            noise.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(d(gen));
            noise.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(d(gen));
            noise.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(d(gen));
        }
    }

    cv::addWeighted(src, 1.0, noise, 1.0, 0.0, dst);
    return dst;
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    cv::Mat noisyImage = addGaussianNoise(image, 0, 30);
    cv::imshow("Noisy Image", noisyImage);
    cv::waitKey(0);
    return 0;
}

```
