# OpenCv

#### 1. 卸载opencv

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

#### 2. 安装

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

- ###### 测试

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



#### 3.查看像素点

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

 

#### 4.多版本管理

- virtualenv

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

- conda

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

- 手动编译

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

3. py

```
import sys
sys.path.append('/usr/local/opencv2/lib/python3.6/site-packages')  # 加载OpenCV2
import cv2 as cv2_2

sys.path.append('/usr/local/opencv3/lib/python3.6/site-packages')  # 加载OpenCV3
import cv2 as cv2_3
```

4. cpp

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

5. cmake+环境变量

```
export OpenCV2_DIR=/usr/local/opencv2
export OpenCV3_DIR=/usr/local/opencv3
```

```
cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)

# 配置OpenCV2
set(OpenCV_DIR $ENV{OpenCV2_DIR}/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(my_project_opencv2 main.cpp)
target_link_libraries(my_project_opencv2 ${OpenCV_LIBS})

# 配置OpenCV3
set(OpenCV_DIR $ENV{OpenCV3_DIR}/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(my_project_opencv3 main.cpp)
target_link_libraries(my_project_opencv3 ${OpenCV_LIBS})
```



- Docker

1. py

```
# Dockerfile for OpenCV 3
FROM python:3.6
RUN pip install opencv-python==3.4.2.17
```

2.cpp

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



```
docker build -t opencv2 -f Dockerfile_opencv2 .
docker build -t opencv3 -f Dockerfile_opencv3 .

docker run -it opencv2 /bin/bash
docker run -it opencv3 /bin/bash
```

## 滤波方法

### 作用：

- 减少噪声，平滑图像--均值 高斯 中值
- 平滑图像，减少细节，柔和图像--边缘检测
- 边缘检测--Sobel滤波，Laplacian滤波，Canny边缘检测
- 特征增强--锐化滤波器--图像边缘细节更清晰
- 形态学操作--腐蚀，膨胀--去斑点，填小孔洞

### 均值滤波
