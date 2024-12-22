# CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(test VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加你需要的配置选项
set(CMAKE_BUILD_TYPE Debug)

# Set the path to your OpenCV 4.10 installation
set(OpenCV_DIR "/opt/opencv-4.10/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
# 设置包含目录-- 头文件路径添加到全局的头文件搜索路径 不够灵活
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    message(STATUS "OpenCV found!")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()
option(OPENCV_ENABLE_NONFREE "Enable non-free algorithms like SIFT" ON)


# 添加包含目录（可选）
# include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)

# target_include_directories(${PROJECT_NAME} PUBLIC 
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  #构建时指定头文件路径。
#         $<INSTALL_INTERFACE:include>                            #安装时指定头文件路径。
#         ${OpenCV_INCLUDE_DIRS}                                  #OpenCV的头文件路径。
# )    

#查找ZBar包
# 查找 zbar 包
#如找不到： export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
#检查： pkg-config --cflags --libs zbar
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBAR REQUIRED zbar)
include_directories(${ZBAR_INCLUDE_DIRS})


# 添加可执行文件
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/test31.cpp)

# 如果有第三方库，可以使用以下方式链接库
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    ${ZBAR_LIBRARIES}
)


