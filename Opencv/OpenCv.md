# OpenCv

## 1. 卸载opencv

```
sudo make uninstall #进入source文件 build
cd ..
sudo rm -r build
sudo rm -r /usr/local/include/opencv2 /usr/local/include/opencv /usr/include/opencv 
 /usr/include/opencv2 /usr/local/share/opencv /usr/local/share/OpenCV /usr/share/opencv 
 /usr/share/OpenCV /usr/local/bin/opencv* /usr/local/lib/libopencv*
 #手动查找
 find /usr iname opencv
 #清空全部
 sudo find / -name "*opencv*" -exec rm -i {} \;
```

## 2. 安装

```
sudo apt-get update 
sudo apt-get upgrade 
sudo apt-get install build-essential libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev #依赖库
sudo apt-get install make
https://opencv.org/releases/ #官网源码
mkdir build && cd ./build
cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=YES -D CMAKE_INSTALL_PREFIX=/opt/opencv-4.10 -D WITH_FFMPEG=ON ..
make -j8
sudo make install
```

- 测试

```
cd /opencv/samples/cpp/example_cmake
cmake .
make
./opencv_example
```

- ###### 环境配置

```
sudo gedit /etc/ld.so.conf.d/opencv.conf #共享库（.so 文件）通常位于标准的库路径下，统在加载库时能够找到这些文件，ldconfig 命令会重新生成动态链接库的缓存，这样新的库路径会被系统识别和使用
/usr/local/lib 保存退出
sudo ldconfig 更新
sudo gedit /etc/bash.bashrc  +
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
source /etc/bash.bashrc
sudo update
sudo gedit /etc/profile + 
export PKG_CONFIG_PATH=/opt/opencv4.10/lib/pkgconfig:$PKG_CONFIG_PATH 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/opencv4.10/lib
source /etc/profile
```

```
/opt/opencv4.10/incldue/opencv4/opencv2
```

```
g++ test.cpp -o test `pkg-config --cflags --libs opencv4`
```

- ###### 扩展库安装

只会为Contrib模块生成新的构建文件

```
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout <your_installed_opencv_version>  # 切换到你已安装的 OpenCV 版本
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout <your_installed_opencv_version>  # 切换到相同的OpenCV版本

cd /path/to/opencv/build  # 或者创建一个新的目录
mkdir build_contrib
cd build_contrib


cmake -D OPENCV_GENERATE_PKGCONFIG=YES -D CMAKE_INSTALL_PREFIX=/opt/opencv-4.10 -D WITH_FFMPEG=ON ..

cmake -DOPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules       -DCMAKE_BUILD_TYPE=Release       -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.10              ../..



      
make -j8
sudo make install

```



## 3.查看像素点

1. 查看某点像素

```
identify -format '%[pixel:p{x,y}]' image.png----jpg路径
```

`x`和`y`替换为你想查看的像素的坐标值。

`R,G,B,A)`，其中R、G、B分别代表红、绿、蓝三种颜色通道的强度，A代表透明度。 

2. 分类级联器cascade

```
Opecv/data/haarcascades
```

## 4.多版本管理

### 包管理结构

#### 1./usr/local/第三方包默认安装

- 可执行文件:默认/usr/local/bin
- 库文件(.so动态.lib静态)：/usr/local/lib
  - /usr/local/lib/cmake/opencv4(cmake配置文件)
- 头文件：/usr/local/incldue/opencv4
- 文档文件：/usr/local/share/opencv4
- 放系统管理员使用的系统管理命令:/usr/sbin
- 指定安装路径

```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/opt/opencv ..
```

#### 2./opt--第三方库，软件包安装目录

```
/opt/
  ├── example/
  │    ├── bin/
  │    ├── lib/
  │    ├── share/
  │    └── ...
  ├── another-software/
  │    ├── bin/
  │    ├── lib/
  │    ├── share/
  │    └── ...
  └── ...
```

- 创建符号链接

```
sudo ln -s /opt/example/bin/example /usr/local/bin/example
```

#### 3 .生成第三方库

- main.cpp -->/bin

```
gcc hello.c -o hello
mv hello /usr/local/bin/
```

- my_lib.c

```
gcc -shared -o libmylib.so -fPIC mylib.c
mv libmylib.so /usr/local/lib/
```

- 调用

```
// main.c
#include <stdio.h>
void greet();
int main() {
    greet();
    return 0;
}
```

- 链接

```
gcc main.c -o main -L/usr/local/lib -lmylib
```

#### 4.命令添加到'PATH'变量

- 临时

```
export PATH=$PATH:/path/to/directory
```

- 永久

```
echo 'export PATH=$PATH:/path/to/directory' >> ~/.bashrc
source ~/.bashrc
```

- 己方.sh直接运行

```
export PATH=$PATH:/usr/local/bin
myscript.sh
```



### virtualenv

```
pip install virtualenv
virtualenv opencv2_env
virtualenv opencv3_env
source opencv2_env/bin/activate
pip install opencv-python==2.4.13.7  # 安装OpenCV 2版本
deactivate

source opencv3_env/bin/activate
pip install opencv-python==3.4.2.17  # 安装OpenCV 3版本
deactivate
```

### conda

```
# 如果使用的是Anaconda，跳过此步骤
conda create -n opencv2_env python=3.6
conda create -n opencv3_env python=3.6

conda activate opencv2_env
conda install -c menpo opencv=2.4.13
conda deactivate

conda activate opencv3_env
conda install -c menpo opencv=3.4.2
conda deactivate
```

### 手动编译

1. 下包，unzip
2. 编译安装

```
cd opencv-2.4.13
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/opencv2 ..
make -j8
sudo make install

cd ../../opencv-3.4.2
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/opencv3 ..
make -j8
sudo make instal
```

## 5.配置

### py

```
import sys
sys.path.append('/usr/local/opencv2/lib/python3.6/site-packages')  # 加载OpenCV2
import cv2 as cv2_2

sys.path.append('/usr/local/opencv3/lib/python3.6/site-packages')  # 加载OpenCV3
import cv2 as cv2_3
```

### cpp

```
cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)

# 添加OpenCV2
set(OpenCV_DIR /usr/local/opencv2/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(my_project_opencv2 main.cpp)
target_link_libraries(my_project_opencv2 ${OpenCV_LIBS})

# 添加OpenCV3
set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(my_project_opencv3 main.cpp)
target_link_libraries(my_project_opencv3 ${OpenCV_LIBS})
```

### cmake+环境变量

```
export OpenCV2_DIR=/usr/local/opencv2
export OpenCV3_DIR=/usr/local/opencv3
```

```
cmake_minimum_required(VERSION 3.1)
project(CheckOpenCV)

# Set the path to your OpenCV 4.10 installation
set(OpenCV_DIR "/opt/opencv-4.10/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(OpenCV_FOUND)
    message(STATUS "OpenCV found!")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

add_executable(CheckOpenCV main.cpp)
target_link_libraries(CheckOpenCV ${OpenCV_LIBS})

```

### Docker

#### py

```
# Dockerfile for OpenCV 3
FROM python:3.6
RUN pip install opencv-python==3.4.2.17
```

#### cpp

```
# Dockerfile for OpenCV 2
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y build-essential cmake git && \
    apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

RUN wget -O opencv-2.4.13.zip https://github.com/opencv/opencv/archive/2.4.13.zip && \
    unzip opencv-2.4.13.zip && \
    cd opencv-2.4.13 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/opencv2 .. && \
    make -j8 && \
    make install

# Dockerfile for OpenCV 3
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y build-essential cmake git && \
    apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

RUN wget -O opencv-3.4.2.zip https://github.com/opencv/opencv/archive/3.4.2.zip && \
    unzip opencv-3.4.2.zip && \
    cd opencv-3.4.2 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/opencv3 .. && \
    make -j8 && \
    make install
```

#### build

```
docker build -t opencv2 -f Dockerfile_opencv2 .
docker build -t opencv3 -f Dockerfile_opencv3 .

docker run -it opencv2 /bin/bash
docker run -it opencv3 /bin/bash
```



# 非极大值抑制实现

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

## 椒盐噪声

- 随机的白色，黑色像素

**cv::Mat dst = src.clone();**    

**std::random_device rd;**    

**std::mt19937 gen(rd());**    

**std::uniform_real_distribution<> d(0, 1);**

指定范围内均匀分布的double数

**double randVal = d(gen);**

```
#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

// Function to add salt and pepper noise to an image
cv::Mat addSaltAndPepperNoise(const cv::Mat &src, double noiseRatio) {
    cv::Mat dst = src.clone();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(0, 1);

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            double randVal = d(gen);
            if (randVal < noiseRatio / 2) {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // Pepper noise
            } else if (randVal < noiseRatio) {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255); // Salt noise
            }
        }
    }
    return dst;
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    cv::Mat noisyImage = addSaltAndPepperNoise(image, 0.02);
    cv::imshow("Noisy Image", noisyImage);
    cv::waitKey(0);
    return 0;
}

```

## 泊松噪声

- 摄像+显微成像-->模拟光子计数

std::random_device rd;    

std::mt19937 gen(rd());

float val = dst.at<cv::Vec3f>(i, j)[c];                

std::poisson_distribution<int> d(val);        

**泊松分布**通常用于描述某个时间段内某个事件发生的次数，其参数是该事件在**单位时间内的平均发生次数**（λ）

dst.at<cv::Vec3f>(i, j)[c] = static_cast<float>(d(gen));



```
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

// Function to add Poisson noise to an image
cv::Mat addPoissonNoise(const cv::Mat &src) {
    cv::Mat dst;
    src.convertTo(dst, CV_32F);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            for (int c = 0; c < dst.channels(); ++c) {
                float val = dst.at<cv::Vec3f>(i, j)[c];
                std::poisson_distribution<int> d(val);
                dst.at<cv::Vec3f>(i, j)[c] = static_cast<float>(d(gen));
            }
        }
    }

    dst.convertTo(dst, src.type());
    return dst;
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    cv::Mat noisyImage = addPoissonNoise(image);
    cv::imshow("Noisy Image", noisyImage);
    cv::waitKey(0);
    return 0;
}

```

## 均匀噪声

每个噪声从均匀分布中独立采样

**std::uniform_real_distribution<> d(-noiseRange, noiseRange);**

均匀分布随机浮点数

**cv::addWeighted(src, 1.0, noise, 1.0, 0.0, dst);**

```
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

// Function to add uniform noise to an image
cv::Mat addUniformNoise(const cv::Mat &src, double noiseRange) {
    cv::Mat noise(src.size(), src.type());
    cv::Mat dst;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(-noiseRange, noiseRange);

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

    cv::Mat noisyImage = addUniformNoise(image, 50);
    cv::imshow("Noisy Image", noisyImage);
    cv::waitKey(0);
    return 0;
}

```



# 滤波方法

## 作用：

- 减少噪声，平滑图像--均值 高斯 中值
- 平滑图像，减少细节，柔和图像--边缘检测
- 边缘检测--Sobel滤波，Laplacian滤波，Canny边缘检测
- 特征增强--锐化滤波器--图像边缘细节更清晰
- 形态学操作--腐蚀，膨胀--去斑点，填小孔洞

## 均值滤波

- 局部邻域内像素平均值-->某点像素值

- 选择一个窗口大小（如 3x3、5x5 等）。
- 对于图像中的每一个像素，以该像素为中心，取出窗口范围内的所有像素值。
- 计算这些像素值的平均值。
- 将该平均值作为窗口中心像素的新值。
- 移动窗口，重复以上步骤，直到处理完整个图像。

**cv::blur(image, result, cv::Size(kernelSize, kernelSize));**

[[1/9 1/9 1/9 ] 

[1/9 1/9 1/9 ]

 [1/9 1/9 1/9 ] ]

模糊细节

```
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 定义输出图像矩阵
    cv::Mat result;

    // 应用均值滤波
    int kernelSize = 5; // 定义卷积核大小
    cv::blur(image, result, cv::Size(kernelSize, kernelSize));

    // 显示原始图像和滤波后的图像
    cv::imshow("Original Image", image);
    cv::imshow("Mean Filtered Image", result);

    // 等待按键
    cv::waitKey(0);
    return 0;
}
```

## 高斯滤波

高斯函数(正态分布)加权平均值来平滑图像--去噪

权重由高斯函数计算得到

高斯滤波器的矩阵元素即为高斯函数在相应位置的值
$$
G(x)= (1/2πσ^2)*
exp(−(x^2+y^2)/2σ^2 )
)
$$
**void cv::GaussianBlur(const cv::Mat& src, cv::Mat& dst, cv::Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT);**

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat gaussianBlur;
    cv::GaussianBlur(image, gaussianBlur, cv::Size(5, 5), 0);
    cv::imshow("Gaussian Blur", gaussianBlur);
    cv::waitKey(0);
    return 0;
}
```



## 中值滤波

取局部邻域像素的中值来去除噪声，对**椒盐**噪声特别有效

**cv::medianBlur(image, medianBlur, 5);**

```
#include <opencv2/opencv.hpp>
int main() {
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat medianBlur;
    cv::medianBlur(image, medianBlur, 5);
    cv::imshow("Median Blur", medianBlur);
    cv::waitKey(0);
    return 0;
}

```

## 双边滤波（高斯改进）

- 像素的空间距离和像素值差异，保留边缘的同时平滑图像

- 双边滤波器通过结合空间域和颜色域来平滑图像。其权重由两个高斯函数的乘积决定：

  - **空间域权重**：考虑像素之间的空间距离，使用一个空间高斯函数。

  - **颜色域权重**：考虑像素之间的颜色差异，使用一个颜色高斯函数

$$
[
I_{\text{new}}(x) = \frac{1}{W_p} \sum_{y \in \Omega} I(y) \cdot \exp \left( -\frac{|x - y|^2}{2\sigma_s^2} \right) \cdot \exp \left( -\frac{|I(x) - I(y)|^2}{2\sigma_r^2} \right)
]
[
W_p = \sum_{y \in \Omega} \exp \left( -\frac{|x - y|^2}{2\sigma_s^2} \right) \cdot \exp \left( -\frac{|I(x) - I(y)|^2}{2\sigma_r^2} \right)
]
$$

**cv::bilateralFilter(image, bilateralBlur, 9, 75, 75);**

第一个参数是输入图像 `image`。

第二个参数是输出图像 `bilateralBlur`。

第三个参数 `9` 是邻域直径（即滤波器的窗口大小）。该值越大，滤波效果越明显。

第四个参数 `75` 是空间高斯函数的标准差 σs\sigma_sσs。

第五个参数 `75` 是颜色高斯函数的标准差 σr\sigma_rσr。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat bilateralBlur;
    cv::bilateralFilter(image, bilateralBlur, 9, 75, 75);
    cv::imshow("Bilateral Blur", bilateralBlur);
    cv::waitKey(0);
    return 0;
}
```



## Sobel 算子 

计算图像的梯度--幅度，进行**边缘检测**

**cv::Sobel(image, sobelX, CV_64F, 1, 0, 5);**

第一个参数是输入图像 `image`。

第二个参数是输出图像 `sobelX`。

第三个参数 `CV_64F` 指定输出图像的深度为 64 位浮点型。使用浮点型是为了防止梯度计算结果溢出。

第四个参数 `1` 指定计算水平梯度（x 方向）。

第五个参数 `0` 指定不计算垂直梯度（y 方向）。

第六个参数 `5` 指定 Sobel 核的大小为 5x5（可选参数，通常为 3）

```
#include <opencv2/opencv.hpp>
int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat sobelX, sobelY;
    cv::Sobel(image, sobelX, CV_64F, 1, 0, 5);
    cv::Sobel(image, sobelY, CV_64F, 0, 1, 5);
    cv::imshow("Sobel X", sobelX);
    cv::imshow("Sobel Y", sobelY);
    cv::waitKey(0);
    return 0;
}
```

### 梯度 

- 通过卷积运算计算图像的梯度，分别计算水平方向（x 方向）和垂直方向（y 方向）的梯度

$$
G_x = I * K_x
$$

$$
G_y = I * K_y
$$



```
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 定义输出图像矩阵
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat grad;

    // 计算 x 方向梯度
    cv::Sobel(image, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // 计算 y 方向梯度
    cv::Sobel(image, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 合并梯度
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Gradient X", abs_grad_x);
    cv::imshow("Gradient Y", abs_grad_y);
    cv::imshow("Combined Gradient", grad);

    // 等待按键
    cv::waitKey(0);
    return 0;
}
```



## Scharr 滤波器(Sobel改进)

- 改进的 Sobel 算子，用于计算图像的梯度
- Scharr 核在 3×3 核的情况下提供了比 Sobel 核更好的性能

**cv::Scharr(image, scharrX, CV_64F, 1, 0);**

第一个参数是输入图像 `image`。

第二个参数是输出图像 `scharrX`。

第三个参数 `CV_64F` 指定输出图像的深度为 64 位浮点型。使用浮点型是为了防止梯度计算结果溢出。

第四个参数 `1` 指定计算水平梯度（x 方向）。

第五个参数 `0` 指定不计算垂直梯度（y 方向）。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat scharrX, scharrY;
    cv::Scharr(image, scharrX, CV_64F, 1, 0);
    cv::Scharr(image, scharrY, CV_64F, 0, 1);
    cv::imshow("Scharr X", scharrX);
    cv::imshow("Scharr Y", scharrY);
    cv::waitKey(0);
    return 0;
}

```

### 梯度

计算过程与 Sobel 算子类似，但使用不同的卷积核

```
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 定义输出图像矩阵
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat grad;

    // 计算 x 方向梯度
    cv::Scharr(image, grad_x, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // 计算 y 方向梯度
    cv::Scharr(image, grad_y, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 合并梯度
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Gradient X", abs_grad_x);
    cv::imshow("Gradient Y", abs_grad_y);
    cv::imshow("Combined Gradient", grad);

    // 等待按键
    cv::waitKey(0);
    return 0;
}

```



##  Laplacian 滤波 

- 增强图像中的边缘细节
- 基于拉普拉斯算子，是一种二阶导数算子，可以检测图像中的边缘
- 计算图像的二阶导数，用于**图像锐化和边缘检测**

**cv::Laplacian(image, laplacian, CV_64F);**

第一个参数是输入图像 `image`。

第二个参数是输出图像 `laplacian`。

第三个参数 `CV_64F` 指定输出图像的深度为 64 位浮点型。使用浮点型是为了防止梯度计算结果溢出。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_64F);
    cv::imshow("Laplacian", laplacian);
    cv::waitKey(0);
    return 0;
}

```

### 梯度

计算图像的二阶导数，可以用于检测图像中的边缘

```
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    // 定义输出图像矩阵
    cv::Mat grad;
    cv::Mat abs_grad;

    // 计算梯度
    cv::Laplacian(image, grad, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad, abs_grad);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Laplacian Gradient", abs_grad);

    // 等待按键
    cv::waitKey(0);
    return 0;
}

```



## Canny 边缘检测 

**多级边缘检测算法**，用于提取图像中的边缘

- **高斯滤波**：首先对图像进行高斯滤波，以减少噪声。
- **梯度计算**：计算图像的梯度强度和方向。
- **非极大值抑制**：对梯度图像进行非极大值抑制，以消除边缘检测中的噪声响应。
- **双阈值检测**：应用双阈值检测来确定强边缘和弱边缘。
- **边缘跟踪**：通过强边缘连接弱边缘，形成最终的边缘图。

注：噪声敏感，用前去噪

**cv::Canny(image, edges, 100, 200);**

`image`：输入的灰度图像。

`edges`：输出的边缘图像。

`100`：低阈值。用于双阈值检测阶段，弱边缘连接强边缘的最低值。

`200`：高阈值。用于双阈值检测阶段，超过此值的边缘将被认为是强边缘。

```
#include <opencv2/opencv.hpp>
int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat edges;
    cv::Canny(image, edges, 100, 200);
    cv::imshow("Canny Edges", edges);
    cv::waitKey(0);
    return 0;
}
```



## 自适应双边滤波（双边改进+平滑自适应）

- 自适应调整参数的双边滤波，去除噪声的同时保留边缘细节

**void cv::adaptiveBilateralFilter(InputArray src, OutputArray dst, Size ksize, double sigmaSpace, double maxSigmaColor = 20.0, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)**

- `src`：输入图像。
- `dst`：输出图像。
- `ksize`：滤波器内核大小，指定窗口尺寸（宽度和高度）。
- `sigmaSpace`：空间域的标准差。它决定了有多远的邻域会影响像素值，越大则影响范围越大。
- `maxSigmaColor`：颜色域的最大标准差，默认值为 20.0。
- `anchor`：表示内核的锚点，默认值为 Point(-1, -1)，表示内核中心。
- `borderType`：边界模式，默认值为 `BORDER_DEFAULT`。

注：**内核大小选择**：内核大小（`ksize`）的选择会影响滤波效果，较大的内核可以平滑更大的噪声区域，但也可能导致边缘模糊。

**参数调整**：`sigmaSpace` 和 `maxSigmaColor` 参数需要根据图像的具体情况进行调整，以达到最佳效果。

**计算复杂度**：自适应双边滤波的计算复杂度较高，适用于需要高质量去噪的应用场景。

```
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
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat dst;
    dst = adaptiveBilateralFilter(image, 9, cv::Size(9, 9), 75, 75);
    cv::imshow("Adaptive Bilateral Blur", adaptiveBilateralBlur);
    cv::waitKey(0);
    return 0;
}
```

## 非局部均值去噪

- 通过相似块的非局部均值去噪
- 彩色图像去噪的非局部均值
- 保留图像的细节和边缘

**void cv::fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hForColorComponents=3, int templateWindowSize=7, int searchWindowSize=21)**

`src`：输入彩色图像（必须是 8 位 3 通道图像）。

`dst`：输出去噪后的图像。

`h`：滤波强度参数。越大则去噪效果越强，但可能会导致图像细节丢失。默认值为 3。

`hForColorComponents`：与 `h` 类似，但应用于颜色通道。默认值为 3。

`templateWindowSize`：用于计算相似度的模板窗口大小。默认值为 7。

`searchWindowSize`：用于搜索相似块的窗口大小。默认值为 21。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat dst;
    cv::fastNlMeansDenoisingColored(image, dst, 10, 10, 7, 21);
    cv::imshow("Non-Local Means Denoising", dst);
    cv::waitKey(0);
    return 0;
}

```

# 特征提取

## 边缘检测

### Canny

### Sobel

  **Normalize the result    cv::normalize(grad, edges, 0, 255, cv::NORM_MINMAX, CV_8U);**

**`src`**：输入数组（源图像）。

**`dst`**：输出数组（目标图像）。

**`alpha`**：归一化后范围的最小值。

**`beta`**：归一化后范围的最大值。

**`norm_type`**：指定归一化类型（例如，`cv::NORM_MINMAX`、`cv::NORM_INF`、`cv::NORM_L1`、`cv::NORM_L2`）。

**`dtype`**：输出数组的类型（默认值为 -1，表示与输入数组类型相同）。

**`mask`**：操作掩码（可选）。

```
#include <opencv2/opencv.hpp>
#include <iostream>

void SobelEdgeDetection(const cv::Mat& src, cv::Mat& edges) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);

    // Compute the gradients in the x and y directions
    cv::Mat grad_x, grad_y;
    cv::Sobel(blurred, grad_x, CV_16S, 1, 0, 3); // Sobel in x direction
    cv::Sobel(blurred, grad_y, CV_16S, 0, 1, 3); // Sobel in y direction

    // Convert gradients to absolute values
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combine the gradients
    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Normalize the result
    cv::normalize(grad, edges, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Perform Sobel edge detection
    cv::Mat edges;
    SobelEdgeDetection(src, edges);

    // Display results
    cv::imshow("Original Image", src);
    cv::imshow("Sobel Edges", edges);
    cv::waitKey(0);

    return 0;
}
```



### Scharr

```
#include <opencv2/opencv.hpp>
#include <iostream>

void ScharrEdgeDetection(const cv::Mat& src, cv::Mat& edges) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);

    // Compute the gradients in the x and y directions using Scharr operator
    cv::Mat grad_x, grad_y;
    cv::Scharr(blurred, grad_x, CV_16S, 1, 0); // Scharr in x direction
    cv::Scharr(blurred, grad_y, CV_16S, 0, 1); // Scharr in y direction

    // Convert gradients to absolute values
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combine the gradients
    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Normalize the result
    cv::normalize(grad, edges, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Perform Scharr edge detection
    cv::Mat edges;
    ScharrEdgeDetection(src, edges);

    // Display results
    cv::imshow("Original Image", src);
    cv::imshow("Scharr Edges", edges);
    cv::waitKey(0);

    return 0;
}

```

### Laplcian

```
#include <opencv2/opencv.hpp>
#include <iostream>

void LaplacianEdgeDetection(const cv::Mat& src, cv::Mat& edges) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);

    // Apply Laplacian operator
    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_16S, 3);

    // Convert to absolute values
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian);

    // Normalize the result
    cv::normalize(abs_laplacian, edges, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Perform Laplacian edge detection
    cv::Mat edges;
    LaplacianEdgeDetection(src, edges);

    // Display results
    cv::imshow("Original Image", src);
    cv::imshow("Laplacian Edges", edges);
    cv::waitKey(0);

    return 0;
}
```

### Prewitt( Sobel变体)

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;

    // Prewitt kernel
    cv::Mat kernel_x = (cv::Mat_<int>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);
    cv::Mat kernel_y = (cv::Mat_<int>(3, 3) <<
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1);

    // 计算 x 方向梯度
    cv::filter2D(image, grad_x, CV_16S, kernel_x);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // 计算 y 方向梯度
    cv::filter2D(image, grad_y, CV_16S, kernel_y);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 合并梯度
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    cv::imshow("Prewitt Edge Detection", grad);
    cv::waitKey(0);
    return 0;
}

```

### Roberts

- 计算图像梯度的算子
- 轻量，注意细节，方向敏感
- 受噪声影响大，边缘定位不准

```
#include <opencv2/opencv.hpp>
#include <iostream>

// Roberts 交叉算子的卷积核
cv::Mat robertsX = (cv::Mat_<char>(2, 2) << 1, 0, 0, -1);
cv::Mat robertsY = (cv::Mat_<char>(2, 2) << 0, 1, -1, 0);

void RobertsEdgeDetection(const cv::Mat& src, cv::Mat& edges) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);

    // Apply Roberts operator
    cv::Mat grad_x, grad_y;
    cv::filter2D(blurred, grad_x, CV_16S, robertsX);
    cv::filter2D(blurred, grad_y, CV_16S, robertsY);

    // Convert gradients to absolute values
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combine the gradients
    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Normalize the result
    cv::normalize(grad, edges, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Perform Roberts edge detection
    cv::Mat edges;
    RobertsEdgeDetection(src, edges);

    // Display results
    cv::imshow("Original Image", src);
    cv::imshow("Roberts Edges", edges);
    cv::waitKey(0);

    return 0;
}

```



## 角点检测

### Harris

- 检测图像中的角点

**cv::cornerHarris(image, dst, blockSize, apertureSize, k);**

- **src**: 输入图像，通常是灰度图像（CV_8U 类型）。
- **dst**: 输出图像，保存 Harris 响应结果，数据类型为 CV_32F。
- **blockSize**: 计算导数自相关矩阵时的邻域大小。这个参数决定了局部区域的大小，用于计算图像梯度的变化。
- **ksize**: Sobel 算子使用的窗口大小（也称为孔径参数）。通常是 3、5 或 7。
- **k**: Harris 角点检测方程中的自由参数，通常取值在 0.04 到 0.06 之间。
- **borderType** (可选): 边界类型，默认值为 `BORDER_DEFAULT`。该参数指定如何处理图像边界，可以选择多种不同的边界处理方式。

**计算方法**

**计算图像梯度**: 使用 Sobel 算子计算图像的 x 和 y 方向梯度。

**计算自相关矩阵**: 对于图像中的每一个像素点，计算其局部区域内的自相关矩阵。

**计算角点响应值**: 根据 Harris 角点检测公式计算角点响应值。

**角点响应图**: 根据响应值生成角点响应图，其中高响应值表示潜在的角点。
$$
R = \det(M) - k \cdot (\text{trace}(M))^2
$$

- M 是图像的自相关矩阵。
- det(M)是矩阵的行列式。
- trace(M) 是矩阵的迹，即对角线元素之和。
- k 是自由参数，通常取值在 0.04 到 0.06 之间。

**cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());**

- **src**: 输入数组（源数组），这里是 `dst`，即 Harris 角点检测的结果。
- **dst**: 输出数组（目标数组），这里是 `dst_norm`，保存规范化后的结果。
- **alpha**: 规范化后数组的最小值，设为 0。
- **beta**: 规范化后数组的最大值，设为 255。
- **norm_type**: 规范化类型，这里是 `cv::NORM_MINMAX`，表示将数组值规范化到 `[alpha, beta]` 范围内。
- **dtype**: 输出数组的类型，这里是 `CV_32FC1`，表示输出数组是单通道的 32 位浮点型。
- **mask**: 可选的操作掩码，这里没有使用，传递一个空的 `cv::Mat`
  - 二值化图像，255--归一化，0 -- 不归一化

**cv::convertScaleAbs(dst_norm, dst_norm_scaled);**

- **src**: 输入数组（源数组），这里是 `dst_norm`，即经过规范化后的数组。
- **dst**: 输出数组（目标数组），这里是 `dst_norm_scaled`，保存缩放和转换后的结果。
- **alpha**: 缩放因子，这里没有设置，使用默认值 1。
- **beta**: 加到每个元素的偏移量，这里没有设置，使用默认值 0。
- -- > u8

**cv::circle(dst_norm_scaled, cv::Point(i, j), 5, cv::Scalar(0), 2, 8, 0);**

- **color**: 圆的颜色（`cv::Scalar`）。这里是 `cv::Scalar(0)`，表示绘制的圆形颜色为黑色。`cv::Scalar` 可以接受多通道颜色值，黑色在灰度图像中对应的是 `0`。
- **thickness**: 圆的边框厚度。如果为负值（如 `-1`），则绘制填充圆。这里是 `2`，表示圆的边框厚度为 2 像素。
- **lineType**: 线条类型。可以是以下几种：
- - `8` 或 `cv::LINE_8`: 8-connected line (默认值)。
  - `4` 或 `cv::LINE_4`: 4-connected line。
  - `cv::LINE_AA`: Anti-aliased line（抗锯齿线）。 这里是 `8`，表示使用 8-connected line。
- **shift**: 圆心坐标和半径的小数位数。这里是 `0`，表示圆心坐标和半径是整数。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(image.size(), CV_32FC1);

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::cornerHarris(image, dst, blockSize, apertureSize, k);

    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > 200) {
                cv::circle(dst_norm_scaled, cv::Point(i, j), 5, cv::Scalar(0), 2, 8, 0);
            }
        }
    }

    cv::imshow("Harris Corners", dst_norm_scaled);
    cv::waitKey(0);
    return 0;
}

```

### Shi-Tomasi（强角点）







**void cv::goodFeaturesToTrack(InputArray image, OutputArray corners, int maxCorners, double qualityLevel, double minDistance, InputArray mask = noArray(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04);**

- **image**: 输入图像，通常是灰度图像（`CV_8U` 类型）。
- **corners**: 输出角点，存储检测到的角点位置，类型为 `std::vector<cv::Point2f>`。
- **maxCorners**: 要检测的最大角点数。如果设置为 0，则表示不限制角点数量，但这可能会导致检测到大量角点。
- **qualityLevel**: 角点的质量水平，通常取值在 0 到 1 之间。这个参数表示角点的最小接受质量。质量水平是相对于最强角点的质量而言的。
- **minDistance**: 角点之间的最小欧几里得距离。如果两个角点之间的距离小于这个值，则其中一个角点将被排除。
- **mask** (可选): 操作掩码，指定在哪些区域内进行角点检测。如果未提供，则使用默认值 `cv::noArray()`，即对整个图像进行检测。
- **blockSize** (可选): 用于计算导数自相关矩阵时的邻域大小。默认值为 3。
- **useHarrisDetector** (可选): 是否使用 Harris 角点检测方法。如果设置为 `true`，则使用 Harris 角点检测方法，否则使用 Shi-Tomasi 角点检测方法。默认值为 `false`。
- **k** (可选): Harris 角点检测方程中的自由参数，仅在 `useHarrisDetector` 设置为 `true` 时使用。默认值为 0.04。



```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int maxCorners = 100;

    cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance);

    for (size_t i = 0; i < corners.size(); i++) {
        cv::circle(image, corners[i], 5, cv::Scalar(255), -1);
    }

    cv::imshow("Shi-Tomasi Corners", image);
    cv::waitKey(0);
    return 0;
}

```

### FAST(快速)

- 一种快速角点检测算法，适用于实时应用

**std::vector<cv::KeyPoint> keypoints;**

- **`pt`**：关键点的坐标。
- **`size`**：关键点邻域的大小（直径）。
- **`angle`**：关键点的方向，取值范围为 [0, 360)。通常由检测算法计算出来，用于描述关键点的主方向。
- **`response`**：关键点的响应值，用于表示关键点的强度或者显著性。响应值越大，表示该点越重要。
- **`octave`**：关键点所在的金字塔层级。图像金字塔是一种图像多分辨率表示方法，用于不同尺度的特征检测。
- **`class_id`**：关键点所属的类别 ID。这个值通常在特定任务中使用，比如对象识别或图像分类。

**cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();**

- **keypoints**: 存储检测到的关键点的向量。每个关键点包含位置、大小、角度、响应值等信息。
- **cv::Ptr[cv::FastFeatureDetector]() fast**: 创建一个 `cv::FastFeatureDetector` 对象的智能指针。`cv::Ptr` 是 OpenCV 中的一种智能指针类型，用于管理对象的生命周期。
- **fast->detect(image, keypoints)**: 使用 FAST 特征检测器在 `image` 图像中检测特征点，并将结果存储在 `keypoints` 向量中。

 **cv::FastFeatureDetector::create(int, bool, cv::FastFeatureDetectorDetectorType);**

- **threshold**: 检测阈值。像素强度变化超过此值时，才被认为是一个角点。默认值通常为 10。
- **nonmaxSuppression**: 是否应用非极大值抑制。设置为 `true` 可以减少角点数量，保留局部区域内响应值最高的角点。默认值为 `true`。
- **type**: FAST 算法类型，可以是 `cv::FastFeatureDetector::TYPE_5_8`, `cv::FastFeatureDetector::TYPE_7_12`, `cv::FastFeatureDetector::TYPE_9_16`，分别表示不同的邻域大小。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image, keypoints);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("FAST Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```



## 特征检测和描述子提取

### SIFT（局部）

- 一种用于检测和描述局部特征的算法

**sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);**

- 该方法同时检测关键点并计算描述符。
- **image**: 输入图像，通常为灰度图像（`CV_8U` 类型）。
- **cv::noArray()**: 表示不使用掩码。掩码可以用于指定图像的某些区域进行特征检测。
- **keypoints**: 输出关键点，类型为 `std::vector<cv::KeyPoint>`。每个关键点包含位置、尺度、方向等信息。
- **descriptors**: 输出描述符，类型为 `cv::Mat`。每个关键点对应一个描述符，描述符是一个向量，表示关键点周围的图像特征。

 **cv::SIFT::create(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6);**

- **nfeatures**: 要保留的特征点的最大数量。默认值为 0，表示不限制特征点数量。
- **nOctaveLayers**: 每个八度中的层数（Scale-space 中的尺度层数）。默认值为 3。
- **contrastThreshold**: 用于过滤弱特征的对比度阈值。默认值为 0.04。
- **edgeThreshold**: 边缘阈值，用于过滤边缘响应强的特征点。默认值为 10。
- **sigma**: 高斯滤波器的标准差，用于初始化图像金字塔。默认值为 1.6。



```
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("SIFT Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### SURF（SIFT加速版）

- SIFT 的一种加速版本，用于快速特征点检测和描述

```
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    surf->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("SURF Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### ORB(快速有效)

- 一种快速且有效的特征点检测和描述算法

**cv::Ptr<cv::ORB> orb = cv::ORB::create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20);**

- **`nfeatures`**: 要保留的特征点的最大数量。默认值为 500。
- **`scaleFactor`**: 图像金字塔的尺度因子。默认值为 1.2f。每一层的尺度为前一层的 `scaleFactor` 倍。
- **`nlevels`**: 金字塔的层数。默认值为 8。
- **`edgeThreshold`**: 边缘阈值，用于过滤边缘响应强的特征点。默认值为 31。
- **`firstLevel`**: 第一个金字塔层的索引。默认值为 0。
- **`WTA_K`**: 用于 BRIEF 描述符的比特数，常设置为 2 或 3。默认值为 2。
- **`scoreType`**: 特征点的评分方法。可以是 `cv::ORB::HARRIS_SCORE` 或 `cv::ORB::FAST_SCORE`。默认值为 `cv::ORB::HARRIS_SCORE`。
- **`patchSize`**: 描述符的大小。默认值为 31。
- **`fastThreshold`**: FAST 特征检测的阈值。默认值为 20。

 **cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);**

**`image`**:

- 输入图像，通常是灰度图像或彩色图像。这个图像将被用作绘制特征点的背景。

**`keypoints`**:

- 特征点的集合，通常是一个 `std::vector<cv::KeyPoint>` 类型的容器。每个 `cv::KeyPoint` 对象包含了特征点的位置、大小、角度等信息。

**`output`**:

- 输出图像，绘制了特征点后的图像。`output` 是一个 `cv::Mat` 类型的矩阵，图像中将显示特征点及其相关信息。

**`cv::Scalar::all(-1)`**:

- 颜色参数，用于指定绘制特征点的颜色。`cv::Scalar::all(-1)` 表示自动选择颜色，即 OpenCV 会根据图像的类型和特征点的位置自动选择颜色。如果希望自定义颜色，可以使用类似 `cv::Scalar(0, 255, 0)`（绿色）这样的参数。

**`cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS`**:

- 标志参数，用于指定绘制特征点的方式。`cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS` 表示绘制带有特征点大小和方向的信息的关键点。
- 其他选项包括 `cv::DrawMatchesFlags::DEFAULT`（默认绘制方式）和 `cv::DrawMatchesFlags::DRAW_OVER_OUTIMG`（将特征点绘制在源图像上）。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("ORB Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### BRIEF(+FAST)

- 一种描述子提取方法，通常与 FAST 角点检测一起使用

**cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();**
**brief->compute(image, keypoints, descriptors);**

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image, keypoints);

    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    brief->compute(image, keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("BRIEF Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### BRISK(快速实时，匹配高效)

- 一种快速、鲁棒的特征点检测和描述子提取算法

- 基于二进制描述符的特征检测方法，结合了特征点检测和描述符生成

**`descriptors`** 是一个 `cv::Mat` 类型的矩阵，存储了每个特征点的描述符。

- **行数**：等于检测到的特征点数量，每行对应一个特征点。
- **列数**：每列对应描述符的一个维度。BRISK 生成的描述符是二进制的，描述符的维度通常为 64 或 128 位。

**cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(int thresh = 30, int octaves = 3, float patternScale = 1.0f);**

- **`thresh`**: 关键点检测的阈值，影响检测到的特征点的数量和质量。默认值为 30。
- **`octaves`**: 图像金字塔的层数，用于不同尺度的特征检测。默认值为 3。
- **`patternScale`**: 用于描述符计算的尺度因子。默认值为 1.0f。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("BRISK Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### FREAK(+FAST二进制)

- 一种基于人眼视网膜模型的二进制描述子提取方法，通常与 FAST 角点检测一起使用

```
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image, keypoints);

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    freak->compute(image, keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("FREAK Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### AKAZE(快速)

- 一种快速的特征点检测和描述子提取算法，适用于实时应用

**cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(**
    **int descriptor_type = cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT,**
    **int descriptor_size = 0,**
    **int descriptor_channels = 3,**
    **int threshold = 0.001f,**
    **int nOctaves = 4,**
    **int nOctaveLayers = 4,**
    **int diffusivity = cv::KAZE_DIFF_PM_G2**
**);**

- **`descriptor_type`**: 描述符类型。`cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT` 和 `cv::AKAZE::DESCRIPTOR_KAZE` 分别表示正立描述符和标准描述符。

- **`descriptor_size`**: 描述符的尺寸。默认值为 0（自动选择）。

- **`descriptor_channels`**: 描述符的通道数。通常为 3（颜色图像）或 1（灰度图像）。

- **`threshold`**: 关键点检测的阈值，影响检测到的特征点的数量和质量。默认值为 0.001f。

  - **作用**: 控制特征点检测的敏感度。`threshold` 是一个浮点数，定义了在特征点检测过程中所使用的尺度空间的阈值。

    **影响**: 较小的阈值会检测到更多的特征点，包括一些可能不那么显著的点，而较大的阈值会检测到更少的特征点，只保留那些显著的点。选择合适的阈值可以平衡特征点的数量和质量。

    **默认值**: 通常默认为 0.001f（具体值可能因 OpenCV 版本而异）。

- **`nOctaves`**: 图像金字塔的层数，用于不同尺度的特征检测。默认值为 4。

  - **`nOctaves`**：
    - **定义**：图像金字塔的层数（octaves）。每个 octave 代表一个图像的分辨率层级，通常是原始图像经过下采样得到的。
    - **默认值**：4。这意味着图像金字塔包含 4 个不同的分辨率层级，从原始图像开始，每个 octave 的图像都比上一个 octave 的图像小一半。
    - **作用**：决定了金字塔的尺度范围。更多的 octaves 可以帮助检测到更大范围的特征，但也会增加计算开销。

- **`nOctaveLayers`**: 每个八度尺度的层数。默认值为 4。

  - **`nOctaveLayers`**：
    - **定义**：每个 octave 内部的尺度层数。每个 octave 包含多个尺度的图像，即在同一分辨率层级下，对图像进行不同程度的模糊处理。
    - **默认值**：4。这意味着每个 octave 内部进行 4 次不同的模糊处理，以捕捉不同尺度的特征。
    - **作用**：决定了每个 octave 中的特征检测精度。更多的尺度层可以帮助更精确地检测特征，但也会增加计算量。

- **`diffusivity`**: 对图像平滑的扩散类型。`cv::KAZE_DIFF_PM_G2` 是平滑扩散的一种类型。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("AKAZE Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### LATCH（+FAST三补丁实时）

- 一种基于三补丁代码的二进制描述子提取方法，适用于实时应用
- 一种基于机器学习的方法，可以为每个关键点计算描述符，通常用于增强特征匹配的准确性

```
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image, keypoints);

    cv::Ptr<cv::xfeatures2d::LATCH> latch = cv::xfeatures2d::LATCH::create();
    latch->compute(image, keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("LATCH Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```

### KAZE(鲁棒特征)

- 一种基于**非线性尺度空间**的特征点检测和描述子提取算法
- **KAZE** 是一种用于检测和描述图像中特征点的算法，与 SIFT 和 SURF 相似，但它使用非线性尺度空间以捕捉更精细的特征。KAZE 特别适用于处理具有较高纹理复杂度和细节的图像。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    kaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    cv::imshow("KAZE Keypoints", output);
    cv::waitKey(0);
    return 0;
}

```



## 描述子匹配

通过一系列特征匹配算法，去除误匹配

### BFMatcher

- 一种简单的暴力匹配方法，用于比较两个描述子集合

**cv::BFMatcher matcher(cv::NORM_HAMMING);**
**std::vector<cv::DMatch> matches;**
**matcher.match(descriptors1, descriptors2, matches);**

- **功能**: 匹配两个图像的描述符，找到最佳匹配的特征点对。

- 参数

  :

  - **`descriptors1`**: 第一个图像的描述符矩阵（由 `sift->detectAndCompute` 生成）。
  - **`descriptors2`**: 第二个图像的描述符矩阵（由 `sift->detectAndCompute` 生成）。
  - **`matches`**: 输出参数，存储匹配结果。`std::vector<cv::DMatch>` 类型，每个 `cv::DMatch` 对象包含一个匹配对的索引和距离信息。

**`void cv::drawMatches(**
    **const cv::Mat& img1,** 
    **const std::vector<cv::KeyPoint>& keypoints1,**
    **const cv::Mat& img2,** 
    **const std::vector<cv::KeyPoint>& keypoints2,**
    **const std::vector<cv::DMatch>& matches,**
    **cv::Mat& outImg,**
    **const cv::Scalar& matchColor = cv::Scalar::all(-1),**
    **const cv::Scalar& singlePointColor = cv::Scalar::all(-1),**
    **const std::vector<char>& matchesMask = std::vector<char>(),**
    **const cv::Scalar& singlePointColor = cv::Scalar::all(-1)**
**);`**

**`img1`**:

- **类型**: `cv::Mat`
- **描述**: 第一个输入图像。特征点从此图像中提取。

**`keypoints1`**:

- **类型**: `std::vector<cv::KeyPoint>`
- **描述**: 第一个图像的关键点。包含所有在 `img1` 中检测到的特征点的信息（位置、大小、方向等）。

**`img2`**:

- **类型**: `cv::Mat`
- **描述**: 第二个输入图像。特征点从此图像中提取。

**`keypoints2`**:

- **类型**: `std::vector<cv::KeyPoint>`
- **描述**: 第二个图像的关键点。包含所有在 `img2` 中检测到的特征点的信息。

**`matches`**:

- **类型**: `std::vector<cv::DMatch>`
- **描述**: 匹配的特征点对。每个 `cv::DMatch` 对象包含一个匹配的索引和距离，表示第一个图像的特征点与第二个图像的特征点之间的匹配关系。

**`outImg`**:

- **类型**: `cv::Mat`
- **描述**: 输出图像，用于显示匹配结果。此图像会包含两张输入图像及其匹配关系的可视化效果。

**`matchColor`** (可选):

- **类型**: `cv::Scalar`
- **描述**: 匹配线的颜色。默认值为 `cv::Scalar::all(-1)`，表示使用随机颜色。如果指定颜色，则所有匹配线将使用该颜色。

**`singlePointColor`** (可选):

- **类型**: `cv::Scalar`
- **描述**: 单个关键点的颜色。默认值为 `cv::Scalar::all(-1)`，表示使用随机颜色。如果指定颜色，则所有关键点将使用该颜色。

**`matchesMask`** (可选):

- **类型**: `std::vector<char>`
- **描述**: 匹配的掩码，表示哪些匹配需要被绘制。掩码的长度与 `matches` 相同，值为 `0` 表示不绘制该匹配，`1` 表示绘制。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error opening images" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output);

    cv::imshow("Matches", output);
    cv::waitKey(0);
    return 0;
}

```

###  FLANN(快速)

- 一种高效的近似最近邻搜索算法，用于快速匹配大规模数据。

**cv::FlannBasedMatcher::FlannBasedMatche**r(const cv::flann::IndexParams& indexParams = cv::flann::KDTreeIndexParams(),**
                                         const cv::flann::SearchParams& searchParams = cv::flann::SearchParams())**

- **`indexParams`**: FLANN 索引参数，用于定义索引构建方式。可以使用不同的 FLANN 索引类型（例如 KDTree）来加速最近邻搜索。
- **`searchParams`**: 搜索参数，用于定义最近邻搜索的行为。例如，设置搜索的近似度和最大迭代次数。

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error opening images" << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    // 绘制匹配结果
    cv::Mat img_matches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches, 
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), std::vector<char>(), 
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 显示匹配结果
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    return 0;
}
```

### RANSAC(鲁棒+单应性变换)

**RANSAC** 是一种鲁棒的参数估计算法，用于从包含**噪声**的数据中**估计模**型。它常用于**过滤匹配中的错误**，并估计**图像配准的变换矩阵**（如单应性矩阵）。

#### **优点**：

- 能够有效处理包含较高比例错误匹配的情况。
- 提供了对模型估计的鲁棒性。

#### 主要步骤

1. **随机选择样本**: 从数据集中随机选择一小部分样本。
2. **模型拟合**: 基于这些样本拟合一个模型。
3. **模型验证**: 使用拟合的模型评估所有数据点，以确定哪些点符合模型（内点）和哪些点不符合（外点）。
4. **重复**: 重复上述步骤多次，每次选择不同的随机样本，直到找到最优的模型。
5. **选择最佳模型**: 选择具有最多内点的模型作为最终模型

#### 应用

通常用于**匹配特征点的几何模型估计**，如计算**单应性矩阵** (homography) 或**基础矩阵** (fundamental matrix)。如何通过特征点匹配计算单应性矩阵。

**读取图像**: 使用 `cv::imread` 读取两张图像。

**转换为灰度图像**: 将图像转换为灰度图像，以便进行特征检测。

**创建 SIFT 特征检测器**: 使用 `cv::SIFT::create()` 创建 SIFT 特征检测器。

**检测和计算**: 对两张图像分别使用 `sift->detectAndCompute` 进行特征点的检测和描述符的计算。

**创建 FLANN 基于的匹配器**: 使用 `cv::FlannBasedMatcher` 创建 FLANN 基于的匹配器。

**匹配描述符**: 使用 `matcher.match` 方法进行特征匹配，得到匹配结果。

**筛选匹配结果**: 使用距离阈值筛选出好的匹配。

**提取匹配点对**: 从匹配结果中提取特征点的坐标。

**计算单应性矩阵**: 使用 `cv::findHomography` 计算单应性矩阵，采用 RANSAC 算法来估计单应性矩阵。

**应用单应性矩阵**: 使用 `cv::warpPerspective` 将第一个图像变换到第二个图像的视角。

**显示结果**: 显示变换后的图像和原图像。

**void cv::warpPerspective(**
    **InputArray src, **  //输入
    **OutputArray dst,**  //输出
    **InputArray M,**  //系数
    **Size dsize, **  //目标图像大小
    **int flags = INTER_LINEAR,**  // 插值方法，计算变换后的像素值	

- `cv::INTER_NEAREST`：最近邻插值
- `cv::INTER_LINEAR`：双线性插值
- `cv::INTER_CUBIC`：三次卷积插值
- `cv::INTER_LANCZOS4`：Lanczos 插值
- 影响变换后图像的质量和速度

​    **int borderMode = BORDER_CONSTANT,** //(边界模式)如何处理变换后图像的边界区域。  `cv::BORDER_CONSTANT`：用常量值填充边界区域（`borderValue`）。

`cv::BORDER_REPLICATE`：复制边界像素。

`cv::BORDER_REFLECT`：反射边界像素。

`cv::BORDER_WRAP`：周期性地填充边界区域。

**const Scalar& borderValue = Scalar()**  //(边界值)

`borderMode` 为 `cv::BORDER_CONSTANT` 时填充边界区域的常量值。**);**



```C++
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // 读取两张输入图像
    cv::Mat image1 = cv::imread("image1.jpg");
    cv::Mat image2 = cv::imread("image2.jpg");
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error opening images" << std::endl;
        return -1;
    }

    // 转换为灰度图像
    cv::Mat gray1, gray2;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    // 创建 SIFT 特征检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 检测关键点并计算描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    // 创建 FLANN 基于的匹配器
    cv::FlannBasedMatcher matcher;

    // 匹配描述符
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 筛选匹配结果，保留前 10% 最佳匹配
    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
    //两描述子向量间的相似性，匹配的可能性
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    // 提取匹配点对
    /*匹配子
    cv::DMatch 结构体包含以下成员变量：
    int queryIdx：表示匹配对中第一个图像中特征点的索引。
    int trainIdx：表示匹配对中第二个图像中特征点的索引。
    int imgIdx：表示图像索引（在当前上下文中通常用不到）。
    float distance：表示两个特征点描述子之间的距离。匹配子
    */
   	/*cv::Point2f pt：特征点的坐标（x, y）。
        float size：特征点的邻域直径。
        float angle：特征点的方向，表示为0到360度之间的角度。
        float response：特征点的响应值，表示特征点的强度。
        int octave：特征点所在的图像金字塔的层级。
        int class_id：特征点的类别，通常用于物体分类任务。
   	*/
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    // 计算单应性矩阵
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    // 应用单应性矩阵进行图像变换 image1->image2
    // target is image2
    cv::Mat result;
    cv::warpPerspective(image1, result, H, image2.size());

    // 显示结果
    cv::imshow("Warped Image", result);
    cv::imshow("Original Image", image2);
    cv::waitKey(0);

    return 0;
}

```

### Descriptor Distance Metrics(距离度量)

**描述符距离度量**--衡量两个特征描述符之间相似度的方

准确匹配和识别特征点至关重要

```
cv::BFMatcher matcher(cv::NORM_HAMMING, true); // 适用于二进制描述符
cv::BFMatcher matcher(cv::NORM_L2); // 欧几里得距离
// 使用 FLANN 基于的匹配器进行匹配
cv::FlannBasedMatcher matcher;
std::vector<cv::DMatch> matches;
matcher.match(descriptors1, descriptors2, matches);
```

### K-Nearest Neighbors (KNN) Matching

**KNN** 匹配算法用于找到每个描述符的 K 个最近邻，以提供更多的匹配候选。

**计算特征描述符**: 对两张图像中的特征点计算描述符。

**计算距离**: 计算描述符之间的距离或相似度。

**选择 K 个最近邻**: 对于每个特征描述符，找到与其最相似的 K 个邻居。

**应用比率测试 (可选)**: 为了进一步过滤匹配结果，通常使用比率测试来去除不良匹配。
$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

$$
\text{if } \text{distance}_{\text{nearest}} < \text{ratio\_threshold} \times \text{distance}_{\text{second\_nearest}}
$$

```
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // 读取两张输入图像
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error opening images" << std::endl;
        return -1;
    }

    // 创建 SIFT 特征检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 检测关键点并计算描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // 创建 KNN 匹配器
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2); // K = 2

    // 使用比率测试来过滤匹配结果
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // 绘制匹配结果
    cv::Mat img_matches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches);
    cv::imshow("KNN Matches", img_matches);
    cv::waitKey(0);

    return 0;
}

```

**读取图像**: 使用 `cv::imread` 读取两张图像，并将其转换为灰度图像。

**检测特征和描述符**: 使用 SIFT 特征检测器对图像进行特征检测，并计算描述符。

**创建 KNN 匹配器**: 使用 `cv::DescriptorMatcher::create` 创建描述符匹配器，指定使用 KNN 算法。

**计算 KNN 匹配**: 使用 `matcher->knnMatch` 计算每个描述符的 K 个最近邻。这里 `K=2`，表示找到每个描述符的两个最近邻。

**比率测试**: 通过比率测试筛选出好的匹配。比率测试通过比较最近邻和次近邻之间的距离来判断匹配是否可靠。

**绘制和显示结果**: 使用 `cv::drawMatches` 绘制匹配结果，并显示在窗口中。

###  **Ratio Test**（KNN升级版）

是一种用于过滤匹配的技术，由 David Lowe 在 SIFT 论文中引入。它基**于 KNN 匹配的结果**，比较**最好的匹配和次好的匹配**的距离比例，以**筛选匹配对**

```
std::vector<cv::DMatch> good_matches;
for (const auto& match : knn_matches) {
    if (match[0].distance < 0.75 * match[1].distance) {
        good_matches.push_back(match[0]);
    }
}
```

###  **Feature Matching with Deep Learning**

- **优点**：
  - 能够处理复杂的场景和变形。
  - 提供了强大的特征表示能力。
- **用法示例**：
  - 使用深度学习框架（如 TensorFlow、PyTorch）训练特征匹配模型。

### 小结

**高级描述子匹配算法**提供了多种工具来提高特征匹配的准确性和效率。结合使用 RANSAC、**KNN**、**比率测试**等技术，可以有效处理图像中的匹配任务

**合适的算法和参数配**------**高效和准确的特征匹配**

# 直方图--特征增强

## 基础

### 计算--单|多通道

bin数：方图中的一个“箱子”或“区间”，用于存储落在该区间内的数据点数量（频率）

**构建**：

- **数据范围**：首先确定数据的总范围。例如，对于灰度图像，数据范围通常是 `[0, 256)`，即像素值从0到255。
- **Bin 数量**：选择将数据范围分割成多少个 bin。例如，将范围 `[0, 256)` 分成256个 bin，每个 bin 对应一个像素值。
- **计算频率**：遍历数据，统计每个 bin 内的数据点数量。直方图的每个 bin 的高度表示该区间内的数据点数量。

**表示：**

- **X 轴**：表示数据范围中的 bin。对于图像直方图，X 轴上的每个 bin 可能表示一个灰度级。
- **Y 轴**：表示每个 bin 的频率，即该 bin 区间内的数据点的数量。

```
cv::calcHist(
    const cv::Mat* images, //图像的指针数组,可以传入多个
    int nimages,  // 图像数量。
    const int* channels, //图像通道索引 0蓝色，1绿色，2红色
    cv::Mat& mask, //可选的掩膜图像 指定感兴趣区域 不需要 空的 cv::Mat 对象
    cv::Mat& hist,d //输出的直方图数据，存储在这里。需要预先分配空间。
    int dims, //直方图的维度 1-3
    const int* histSize,//每个维度的直方图的大小（bin数目）。一个数组，每个元素表示直方图在该维度上的大小。
    const float** ranges,//每个维度的取值范围。灰度图像，范围常[0, 256)。这是一个指向浮点数组的指针，其中每个维度都有两个值：最小值和最大值。
    bool accumulate = false//是否在计算直方图时累加到现有的直方图。如果为 true，则计算结果将累加到已存在的直方图中。如果为 false，则覆盖现有的直方图
);
```

```
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE); // 读取灰度图像
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 计算直方图
    int histSize = 256; // 灰度图像的灰度级数
    float range[] = {0, 256}; // 像素值的范围
    const float* histRange = {range}; // 直方图范围
    cv::Mat hist;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // 打印直方图的前几个值
    for (int i = 0; i < histSize; ++i) {
        std::cout << hist.at<float>(i) << std::endl;
    }

    return 0;
}
```

### 绘制直方图

**cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);**

- 输入 输出 min max 

- `cv::NORM_MINMAX`：将数据线性地映射到指定的最小值和最大值之间。

  `cv::NORM_L1`、`cv::NORM_L2`：用于不同的归一化标准，如 L1 范数或 L2 范数，通常用于其他类型的矩阵归一化

```
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;

    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // 归一化直方图
    cv::Mat histImage(histSize, histSize, CV_8UC1, cv::Scalar(0));
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

    // 绘制直方图
    for (int i = 1; i < histSize; ++i) {
        cv::line(histImage, cv::Point(i - 1, histSize - cvRound(hist.at<float>(i - 1))),
                 cv::Point(i, histSize - cvRound(hist.at<float>(i))),
                 cv::Scalar(255), 2, 8, 0);
    }

    cv::imshow("Histogram", histImage);
    cv::waitKey(0);

    return 0;
}

```

### 直方图均衡化 

- 增强图像对比度，使得图像的**暗部**和**亮部**更加明显

```
void cv::equalizeHist(
    const cv::Mat& src,       // 输入图像 须是单通道的灰度 适用于灰度
    cv::Mat& dst              // 输出图像，均衡化后的结果，相同的尺寸和类型。
);
/*原理：
计算累积分布函数 (CDF)：

计算输入图像的灰度直方图。
根据直方图计算累积分布函数 (CDF)，CDF 描述了每个灰度级的累积频率。
计算映射关系：

计算灰度级映射关系，使得图像的直方图分布更均匀。
使用 CDF 计算每个灰度级的均衡化值。
应用映射：

将映射应用到输入图像，生成均衡化后的图像。
*/
```



```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);

    cv::imshow("Original Image", image);
    cv::imshow("Equalized Image", equalizedImage);
    cv::waitKey(0);

    return 0;
}

```

### **计算归一化直方图**

```
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;

    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    cv::Mat histImage(histSize, histSize, CV_8UC1, cv::Scalar(0));

    // 归一化
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; ++i) {
        cv::line(histImage, cv::Point(i - 1, histSize - cvRound(hist.at<float>(i - 1))),
                 cv::Point(i, histSize - cvRound(hist.at<float>(i))),
                 cv::Scalar(255), 2, 8, 0);
    }

    cv::imshow("Normalized Histogram", histImage);
    cv::waitKey(0);

    return 0;
}

```

### 计算二维直方图--显示彩图颜色分布

```
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

    // 绘制直方图等操作
    // ... (类似于灰度图像直方图的绘制)

    return 0;
}

```

# 白光平衡

## 目的

调整图像的色温，使图像中的白色和灰色看起来真实-->

消除光(日光 白炽 荧光)源的色偏，使图像的颜色更接近于人眼感知的自然颜色

## 类别

**自动白光平衡（AWB）**：

- 相机或图像处理软件自动调整图像的色温。
- 通常基于图像中某些区域的色彩分布来进行调整。

**手动白光平衡**：

- 用户手动设置色温参数。
- 适用于特定的拍摄条件或需要精确控制颜色的情况。

## 实现

### 灰度世界假设

假设灰度世界色素的色温是中性的(R G B 三通道均值相等)

```
#include <opencv2/opencv.hpp>

cv::Mat whiteBalanceGrayWorld(const cv::Mat& image) {
    cv::Mat result;
    cv::Mat floatImg;
    image.convertTo(floatImg, CV_32FC3);

    // 计算每个通道的均值
    cv::Scalar mean = cv::mean(floatImg);
    cv::Scalar meanB = mean[0], meanG = mean[1], meanR = mean[2];

    // 计算调整系数
    float avg = (meanB + meanG + meanR) / 3;
    float scaleB = avg / meanB;
    float scaleG = avg / meanG;
    float scaleR = avg / meanR;

    // 调整每个通道
    std::vector<cv::Mat> channels;
    cv::split(floatImg, channels);
    channels[0] *= scaleB;
    channels[1] *= scaleG;
    channels[2] *= scaleR;
    cv::merge(channels, result);

    // 转换回 8 位图像
    result.convertTo(result, CV_8UC3);
    return result;
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat balancedImage = whiteBalanceGrayWorld(image);

    cv::imshow("Original Image", image);
    cv::imshow("White Balanced Image", balancedImage);
    cv::waitKey(0);

    return 0;
}

```

### 白点校正

假设图像中某个区域白色| 接近白色

```
#include <opencv2/opencv.hpp>

cv::Mat whiteBalanceWhitePatch(const cv::Mat& image) {
    cv::Mat result;
    cv::Mat floatImg;
    image.convertTo(floatImg, CV_32FC3);

    // 假设右下角是白色区域
    cv::Rect whitePatch = cv::Rect(image.cols - 50, image.rows - 50, 50, 50);
    cv::Mat whitePatchROI = floatImg(whitePatch);

    // 计算白点的最大值
    cv::Scalar maxVal = cv::max(whitePatchROI);

    // 计算调整系数
    cv::Scalar avgVal = cv::mean(maxVal);
    float scaleB = 255 / avgVal[0];
    float scaleG = 255 / avgVal[1];
    float scaleR = 255 / avgVal[2];

    // 调整每个通道
    std::vector<cv::Mat> channels;
    cv::split(floatImg, channels);
    channels[0] *= scaleB;
    channels[1] *= scaleG;
    channels[2] *= scaleR;
    cv::merge(channels, result);

    // 转换回 8 位图像
    result.convertTo(result, CV_8UC3);
    return result;
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat balancedImage = whiteBalanceWhitePatch(image);

    cv::imshow("Original Image", image);
    cv::imshow("White Balanced Image", balancedImage);
    cv::waitKey(0);

    return 0;
}

```

### 色温调整

```
void cv::createTrackbar(
    const std::string& trackbarName,    // 滑块名称
    const std::string& windowName,      // 窗口名称
    int* value,                         // 滑块的初始值和回调函数参数
    int maxValue,                       // 滑块的最大值
    void(*onChange)(int, void*) = 0,    // 滑块值变化的回调函数
    void* userData = 0                 // 传递给回调函数的用户数据
);
```

手动调整色温参数

```
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat adjustWhiteBalance(cv::Mat image, double redGain, double greenGain, double blueGain) {
    cv::Mat result;
    cv::Mat floatImg;

    // 将图像转换为浮点型以进行精确处理
    image.convertTo(floatImg, CV_32FC3);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(floatImg, channels);

    // 调整每个通道的增益
    channels[0] *= blueGain; // 蓝色通道
    channels[1] *= greenGain; // 绿色通道
    channels[2] *= redGain; // 红色通道

    // 合并通道
    cv::merge(channels, result);

    // 确保像素值在 [0, 255] 范围内
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8UC3);

    return result;
}

void onTrackbarChange(int, void* userData) {
    auto* data = reinterpret_cast<std::tuple<cv::Mat*, double*, double*, double*>*>(userData);
    cv::Mat* image = std::get<0>(*data);
    double* redGain = std::get<1>(*data);
    double* greenGain = std::get<2>(*data);
    double* blueGain = std::get<3>(*data);

    // 获取滑块值
    int redValue = cv::getTrackbarPos("Red Gain", "White Balance");
    int greenValue = cv::getTrackbarPos("Green Gain", "White Balance");
    int blueValue = cv::getTrackbarPos("Blue Gain", "White Balance");

    // 计算增益系数
    *redGain = 1.0 + (redValue - 50) / 50.0;
    *greenGain = 1.0 + (greenValue - 50) / 50.0;
    *blueGain = 1.0 + (blueValue - 50) / 50.0;

    // 调整白光平衡
    cv::Mat balancedImage = adjustWhiteBalance(*image, *redGain, *greenGain, *blueGain);

    // 显示图像
    cv::imshow("White Balanced Image", balancedImage);
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    double redGain = 1.2;
    double greenGain = 1.0;
    double blueGain = 1.1;

    // 创建窗口
    cv::namedWindow("White Balance");

    // 创建滑块
    cv::createTrackbar("Red Gain", "White Balance", nullptr, 100, onTrackbarChange,
                       new std::tuple<cv::Mat*, double*, double*, double*>(&image, &redGain, &greenGain, &blueGain));
    cv::createTrackbar("Green Gain", "White Balance", nullptr, 100, onTrackbarChange,
                       new std::tuple<cv::Mat*, double*, double*, double*>(&image, &redGain, &greenGain, &blueGain));
    cv::createTrackbar("Blue Gain", "White Balance", nullptr, 100, onTrackbarChange,
                       new std::tuple<cv::Mat*, double*, double*, double*>(&image, &redGain, &greenGain, &blueGain));

    // 初始化滑块位置
    cv::setTrackbarPos("Red Gain", "White Balance", static_cast<int>(redGain * 50 - 50));
    cv::setTrackbarPos("Green Gain", "White Balance", static_cast<int>(greenGain * 50 - 50));
    cv::setTrackbarPos("Blue Gain", "White Balance", static_cast<int>(blueGain * 50 - 50));

    // 显示初始图像
    onTrackbarChange(0, new std::tuple<cv::Mat*, double*, double*, double*>(&image, &redGain, &greenGain, &blueGain));

    cv::waitKey(0);

    return 0;
}

```

### 色彩空间转换

RBG --> lab 调整L通道值，返回RBG空间  -- 处理复杂光照

```
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat whiteBalanceUsingLab(cv::Mat image, double scale) {
    cv::Mat labImage;
    cv::Mat result;

    // 转换到 Lab 色彩空间
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // 分离 L, a, b 通道
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // 调整 L 通道
    labChannels[0] *= scale;

    // 合并通道
    cv::merge(labChannels, labImage);

    // 转换回 BGR 色彩空间
    cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);

    // 确保像素值在 [0, 255] 范围内
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8UC3);

    return result;
}

void onTrackbarChange(int, void* userData) {
    auto* data = reinterpret_cast<std::tuple<cv::Mat*, double*>*>(userData);
    cv::Mat* image = std::get<0>(*data);
    double* scale = std::get<1>(*data);

    // 获取滑块值并计算增益系数
    int trackbarValue = cv::getTrackbarPos("L Channel Scale", "White Balance");
    *scale = 0.5 + (trackbarValue / 50.0); // 增益范围从 0.5 到 2.0

    // 调整白光平衡
    cv::Mat balancedImage = whiteBalanceUsingLab(*image, *scale);

    // 显示图像
    cv::imshow("White Balanced Image", balancedImage);
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    double scale = 1.0; // 初始增益系数

    // 创建窗口
    cv::namedWindow("White Balance");

    // 创建滑块
    cv::createTrackbar("L Channel Scale", "White Balance", nullptr, 100, onTrackbarChange,
                       new std::tuple<cv::Mat*, double*>(&image, &scale));

    // 初始化滑块位置
    cv::setTrackbarPos("L Channel Scale", "White Balance", static_cast<int>(scale * 50.0));

    // 显示初始图像
    onTrackbarChange(0, new std::tuple<cv::Mat*, double*>(&image, &scale));

    cv::waitKey(0);

    return 0;
}

```

### 高级自实现

#### 简单调整

```
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 创建白平衡调整器
    cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();

    cv::Mat whiteBalancedImage;
    wb->balanceWhite(image, whiteBalancedImage);

    cv::imshow("Original Image", image);
    cv::imshow("White Balanced Image", whiteBalancedImage);
    cv::waitKey(0);

    return 0;
}

```

#### 灰度假设

```
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>

// 函数：调整图像亮度（示例用来调整滑块参数）
cv::Mat adjustBrightness(cv::Mat image, double alpha) {
    cv::Mat result;
    image.convertTo(result, -1, alpha, 0);// * +
    return result;
}

// 回调函数：当滑块值变化时调用
void onTrackbarChange(int, void* userData) {
    auto* data = reinterpret_cast<std::tuple<cv::Mat*, cv::Ptr<cv::xphoto::GrayworldWB>, double*>*>(userData);
    cv::Mat* image = std::get<0>(*data);
    cv::Ptr<cv::xphoto::GrayworldWB> wb = std::get<1>(*data);
    double* alpha = std::get<2>(*data);

    // 获取滑块值
    int trackbarValue = cv::getTrackbarPos("Brightness", "White Balance");

    // 计算亮度调整系数
    *alpha = 1.0 + (trackbarValue - 50) / 50.0; // 亮度范围从 0.5 到 1.5

    // 调整亮度
    cv::Mat brightImage = adjustBrightness(*image, *alpha);

    // 白平衡调整
    cv::Mat whiteBalancedImage;
    wb->balanceWhite(brightImage, whiteBalancedImage);

    // 显示图像
    cv::imshow("White Balanced Image", whiteBalancedImage);
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    double alpha = 1.0; // 初始亮度调整系数

    // 创建灰世界白平衡调整器
    cv::Ptr<cv::xphoto::GrayworldWB> wb = cv::xphoto::createGrayworldWB();

    // 创建窗口
    cv::namedWindow("White Balance");

    // 创建滑块
    cv::createTrackbar("Brightness", "White Balance", nullptr, 100, onTrackbarChange,
                       new std::tuple<cv::Mat*, cv::Ptr<cv::xphoto::GrayworldWB>, double*>(&image, wb, &alpha));

    // 初始化滑块位置
    cv::setTrackbarPos("Brightness", "White Balance", static_cast<int>(alpha * 50.0));

    // 显示初始图像
    onTrackbarChange(0, new std::tuple<cv::Mat*, cv::Ptr<cv::xphoto::GrayworldWB>, double*>(&image, wb, &alpha));

    cv::waitKey(0);

    return 0;
}

```

#### 基于学习

```
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 创建基于学习的白平衡调整器
    cv::Ptr<cv::xphoto::LearningBasedWB> wb = cv::xphoto::createLearningBasedWB();

    cv::Mat whiteBalancedImage;
    wb->balanceWhite(image, whiteBalancedImage);

    cv::imshow("Original Image", image);
    cv::imshow("White Balanced Image", whiteBalancedImage);
    cv::waitKey(0);

    return 0;
}

```

# 图像分割

将图像**分成多个区域**或对象，提取ROI

## 阈值分割

### 全局阈值

**描述**: 基于全局阈值将图像分成前景和背景。`127` 是阈值，`255` 是前景像素的值。

**`cv::THRESH_BINARY`**：将大于等于阈值的像素值设置为 `maxValue`，小于阈值的像素值设置为 0。

**`cv::THRESH_BINARY_INV`**：将大于等于阈值的像素值设置为 0，小于阈值的像素值设置为 `maxValue`。

**`cv::THRESH_TRUNC`**：将大于等于阈值的像素值截断为阈值，小于阈值的像素值保持不变。

**`cv::THRESH_TOZERO`**：将小于阈值的像素值设置为 0，大于等于阈值的像素值保持不变。

**`cv::THRESH_TOZERO_INV`**：将大于等于阈值的像素值设置为 0，小于阈值的像素值保持不变。

```
cv::Mat gray;
cv::Mat binary;
cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
```

### 自适应阈值 

**描述**: 计算图像局部区域的阈值，以适应不同的光照条件。使用局部均值和高斯加权来动态调整阈值。

`void cv::adaptiveThreshold(`

   ` const cv::Mat& src,  `      // 输入图像（灰度图像）

 `   cv::Mat& dst,     `         // 输出图像（经过自适应阈值处理后的二值图像）

  `  double maxValue,    `       // 阈值处理后的最大值

   ` int adaptiveMethod,      `  // 自适应阈值方法 {

**`cv::ADAPTIVE_THRESH_MEAN_C`**：使用邻域块内的平均值作为阈值。

**`cv::ADAPTIVE_THRESH_GAUSSIAN_C`**：使用邻域块内的加权平均值（高斯权重）作为阈值。G(x,y)

}

​    `int thresholdType,   `      // 阈值类型{

**`cv::THRESH_BINARY`**：将阈值处理后的像素值设置为 `maxValue`，小于阈值的像素值设置为 0。

**`cv::THRESH_BINARY_INV`**：将阈值处理后的像素值设置为 0，大于等于阈值的像素值设置为 `maxValue`。

}

​    int blockSize,       `      // 邻域块的大小   奇数 {3, 5, 7 }

`    double C    `               // 常数，用于调整阈值{减去这个值来调整阈值}

`);`

```
cv::Mat binary;
cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
```

## 边缘检测

## 区域生长

手动实现，一般包括  选择种子点和生长区域。

- 选点：阈值化 | 边缘检测  | 自动选择

- 定义生长条件：断邻域像素是否应包含在当前区域的条件

  - **灰度值差异**：邻域像素的灰度值与种子点的灰度值之间的差异。

    **颜色差异**：在彩色图像中，邻域像素的颜色与种子点的颜色之间的差异。

    **纹理特征**：图像的纹理特征与种子点的纹理特征之间的差异。

- 区域生长

  - 从种子点开始，检查其邻域像素是否满足生长条件。

    如果满足条件，则将该像素加入当前区域，并将该像素的邻域像素作为待检查的候选像素。

    重复这一过程，直到所有满足条件的像素都被包括在内，且没有更多的像素可以加入。

- 停止条件：生长过程通常在以下条件下停止：

  - 区域达到最大预设大小。
  - 没有更多的邻域像素满足生长条件。
  - 所有种子点都被处理完毕。

- 优点：出来复杂 灰度接近图形

- 缺点 ： 噪声敏感 手动种植

```
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>

void regionGrowing(const cv::Mat& src, cv::Mat& dst, cv::Point seed, int threshold) {
    // 初始化输出图像
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // 获取图像的尺寸
    int rows = src.rows;
    int cols = src.cols;

    // 定义四邻域（上下左右）
    std::vector<cv::Point> neighbors = { cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, -1), cv::Point(0, 1) };

    // 队列用于存储待处理的像素
    std::queue<cv::Point> queue;
    queue.push(seed);

    // 获取种子点的灰度值
    uchar seedValue = src.at<uchar>(seed);

    while (!queue.empty()) {
        cv::Point p = queue.front();
        queue.pop();

        // 如果像素已经在区域内或超出图像边界，则跳过
        if (dst.at<uchar>(p) != 0 || p.x < 0 || p.y < 0 || p.x >= cols || p.y >= rows)
            continue;

        // 获取当前像素的灰度值
        uchar currentValue = src.at<uchar>(p);
        if (std::abs(currentValue - seedValue) <= threshold) {
            // 将像素添加到区域内
            dst.at<uchar>(p) = 255;

            // 将邻域像素添加到队列
            for (const auto& neighbor : neighbors) {
                cv::Point neighborPoint = p + neighbor;
                queue.push(neighborPoint);
            }
        }
    }
}

int main() {
    // 读取灰度图像
    cv::Mat gray = cv::imread("example.jpg", cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // 创建一个用于存储分割结果的矩阵
    cv::Mat segmented;

    // 选择种子点
    cv::Point seed(100, 100);  // 示例种子点位置
    int threshold = 20;  // 灰度差异阈值

    // 执行区域生长分割
    regionGrowing(gray, segmented, seed, threshold);

    // 显示原始图像和分割结果
    cv::imshow("Gray Image", gray);
    cv::imshow("Segmented Image", segmented);

    cv::waitKey(0);
    return 0;
}

```

##  **图割 (Graph Cut)**

```
cpp复制代码cv::Mat mask, bgdModel, fgdModel;
cv::grabCut(image, mask, cv::Rect(50, 50, 400, 400), bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
cv::Mat result;
image.copyTo(result, mask);
```

- **描述**: 基于图论的方法，将图像视作一个图，并通过最小化能量函数进行分割。

## **水平集方法 (Level Set Method)**

```
cpp
复制代码
// 水平集方法通常需要结合特定的库或工具包进行实现。
```

- **描述**: 使用水平集函数对图像进行分割，常用于医学图像处理。

## **K-means 聚类 (K-means Clustering)**

```
cpp复制代码cv::Mat samples, labels, centers;
cv::Mat samples_reshaped = image.reshape(1, image.rows * image.cols);
samples_reshaped.convertTo(samples, CV_32F);
cv::kmeans(samples, 3, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2), 3, cv::KMEANS_PP_CENTERS, centers);
cv::Mat segmented_image(image.size(), CV_8UC3);
for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
        segmented_image.at<cv::Vec3b>(i, j) = centers.at<cv::Vec3f>(labels.at<int>(i * image.cols + j), 0);
    }
}
```

- **描述**: 使用 K-means 聚类算法将图像分成指定数量的簇，常用于颜色分割。

## **图像分水岭变换 (Watershed Transformation)**

```
cpp复制代码cv::Mat gray, binary, dist_transform, dist_transform_8u;
cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 3);
cv::normalize(dist_transform, dist_transform_8u, 0, 255, cv::NORM_MINMAX);
cv::Mat markers;
cv::threshold(dist_transform, markers, 0.7 * cv::norm(dist_transform, cv::NORM_L2), 255, cv::THRESH_BINARY);
cv::watershed(image, markers);
```

- **描述**: 基于拓扑学方法对图像进行分割，模拟水流在地形上的流动。

## **分水岭变换 (Watershed Transformation)**

```
cpp复制代码cv::Mat markers;
cv::watershed(image, markers);
```

- **描述**: 使用分水岭算法对图像进行分割。图像中的每个区域被看作“山脊”，分水岭变换用于将这些区域分开。

##  **深度学习方法 (Deep Learning Methods)**

#### **图像分割网络 (Image Segmentation Networks)**

- **U-Net**: 适用于医学图像分割。
- **Mask R-CNN**: 用于实例分割。
- **DeepLab**: 用于语义分割。

```
cpp
复制代码
// 深度学习方法通常需要使用如 TensorFlow、PyTorch 等深度学习框架。
```

## **分水岭变换 (Watershed Transformation)**

- **描述**: 使用分水岭算法对图像进行分割。图像中的每个区域被看作“山脊”，分水岭变换用于将这些区域分开。
