#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/*
标准pnp算法：
0    SOLVEPNP_ITERATIVE  //最少4点 传统迭代 少量点 一般应用 稳定 
1    SOLVEPNP_EPNP   // 最少4点 大数据 高效 精度低
2    SOLVEPNP_P3P    // 3个点 快速
3    SOLVEPNP_DLS    // 最少6点 多个点 噪声敏感 稳定性差
4    SOLVEPNP_UPNP   // 最少4点 非迭代 点数大于4 既定几何形状
5    SOLVEPNP_AP3P   // 3个点 改进 高精度
8    SOLVEPNP_SQPNP  // 最少4点 高进度 高效
共面pnp算法：
6    SOLVEPNP_IPPE  //最少4点 高精度 小误差 更稳定 平面标定 二维码识别
7    SOLVEPNP_IPPE_SQUARE // 最少4点 精度高 平面标定 二维码识别 -- 正方 | 矩形
*/

//旋转矩阵：相机在世界坐标系下的旋转矩阵
//平移矩阵：相机在世界坐标系下的坐标
// Pcamera = R * Pworld - T  -- 可解算出相机坐标系下的点坐标
// Pworld = RT * Pcamera + RT * T


// 图像2D ： 相机3D  K--内参矩阵  Z--深度
// 构造齐次: P2Dh = [x, y, 1]
// 反投影: Pcamera = Z * KT * P2Dh
// 解算: Pworld = RT * Pcamera + RT * T  --- 没必要
// 深度估计： H--实际高度 h--像素高度 d--小孔到成像面距离(焦距)  
//     Z = H*d/h


// PnP算法封装类
class PnPHandler {
public:
    PnPHandler(const vector<Point3f>& objectPoints, const vector<Point2f>& imagePoints, const Mat& cameraMatrix, const Mat& distCoeffs)
        : objectPoints(objectPoints), imagePoints(imagePoints), cameraMatrix(cameraMatrix), distCoeffs(distCoeffs) {}

    // 设置PnP算法类型
    void setAlgorithm(int method) {
        pnpMethod = method;
    }

    // 计算相机位姿
    void estimatePose(Mat& rvec, Mat& tvec) {
        // 调用OpenCV PnP函数
        solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, pnpMethod);
    }

private:
    vector<Point3f> objectPoints;  // 3D点
    vector<Point2f> imagePoints;    // 2D点
    Mat cameraMatrix;               // 相机内参
    Mat distCoeffs;                 // 畸变系数
    int pnpMethod = SOLVEPNP_ITERATIVE; // 默认PnP算法
};

int main() {
    // 定义3D物体点（例如，正方体的角点）
    vector<Point3f> objectPoints = {
        Point3f(0.0f, 0.0f, 0.0f),
        Point3f(1.0f, 0.0f, 0.0f),
        Point3f(1.0f, 1.0f, 0.0f),
        Point3f(0.0f, 1.0f, 0.0f)
    };

    // 定义对应的2D图像点
    vector<Point2f> imagePoints = {
        Point2f(320.0f, 240.0f),
        Point2f(400.0f, 240.0f),
        Point2f(400.0f, 320.0f),
        Point2f(320.0f, 320.0f)
    };

    // 定义相机内参（示例值）
    Mat cameraMatrix = (Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F); // 假设没有畸变

    // 创建PnP处理对象
    PnPHandler pnpHandler(objectPoints, imagePoints, cameraMatrix, distCoeffs);

    // 选择PnP算法
    pnpHandler.setAlgorithm(SOLVEPNP_EPNP); // 可以选择其他算法

    // 估计相机位姿
    Mat rvec, tvec;
    pnpHandler.estimatePose(rvec, tvec);

    // 输出结果
    cout << "Rotation Vector:" << endl << rvec << endl;
    cout << "Translation Vector:" << endl << tvec << endl;

    return 0;
}
