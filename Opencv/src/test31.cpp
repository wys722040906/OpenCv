#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class ImageDeblur {
private:
    // 运动去模糊参数
    const int MOTION_KERNEL_SIZE = 15;  // 增大核大小以处理更严重的运动模糊
    const double ANGLE_STEP = 15.0;     // 角度步进
    const int NUM_ANGLES = 24;          // 360/15 = 24个角度
    
public:
    Mat processFrame(const Mat& frame) {
        Mat result;
        frame.copyTo(result);
        
        // 步骤1: 运动方向估计和去模糊
        Mat best_deblurred;
        double best_variance = 0;
        
        // 对不同角度进行去模糊尝试
        for(int i = 0; i < NUM_ANGLES; i++) {
            double angle = i * ANGLE_STEP;
            Mat motion_kernel = getMotionKernel(MOTION_KERNEL_SIZE, angle);
            
            Mat deblurred;
            deconvolution(frame, deblurred, motion_kernel);
            
            // 计算清晰度评估指标
            double variance = calculateVariance(deblurred);
            
            if(variance > best_variance) {
                best_variance = variance;
                deblurred.copyTo(best_deblurred);
            }
        }
        
        // 步骤2: 自适应锐化
        Mat sharpened;
        adaptiveSharpening(best_deblurred, sharpened);
        
        // 步骤3: 去噪同时保持边缘
        bilateralFilter(sharpened, result, 9, 75, 75);
        
        return result;
    }

private:
    // 生成运动模糊核
    Mat getMotionKernel(int size, double angle) {
        Mat kernel = Mat::zeros(size, size, CV_32F);
        Point center(size/2, size/2);
        double rad = angle * CV_PI / 180.0;
        double cos_val = cos(rad);
        double sin_val = sin(rad);
        
        // 在核上画一条线来模拟运动模糊
        Point2f direction(cos_val, sin_val);
        line(kernel, 
             Point(center.x - direction.x * size/2, center.y - direction.y * size/2),
             Point(center.x + direction.x * size/2, center.y + direction.y * size/2),
             1.0, 1);
        
        kernel = kernel / sum(kernel);
        return kernel;
    }
    
    // 使用Richardson-Lucy算法进行去卷积
    void deconvolution(const Mat& src, Mat& dst, const Mat& kernel, int iterations = 5) {
        Mat float_src;
        src.convertTo(float_src, CV_32F);
        
        // 初始估计
        float_src.copyTo(dst);
        
        Mat kernel_flip;
        flip(kernel, kernel_flip, -1);
        
        for(int i = 0; i < iterations; i++) {
            Mat conv;
            filter2D(dst, conv, -1, kernel);
            
            Mat ratio;
            divide(float_src, conv + 1e-10, ratio);
            
            Mat est;
            filter2D(ratio, est, -1, kernel_flip);
            
            multiply(dst, est, dst);
        }
        
        dst.convertTo(dst, CV_8U);
    }
    
    // 计算图像方差作为清晰度度量
    double calculateVariance(const Mat& img) {
        Mat gray;
        if(img.channels() > 1)
            cvtColor(img, gray, COLOR_BGR2GRAY);
        else
            img.copyTo(gray);
        
        Scalar mean, stddev;
        meanStdDev(gray, mean, stddev);
        
        return stddev[0] * stddev[0];
    }
    
    // 自适应锐化
    void adaptiveSharpening(const Mat& src, Mat& dst) {
        Mat laplacian;
        Laplacian(src, laplacian, CV_32F, 3);
        
        // 计算局部方差来自适应调整锐化强度
        Mat variance;
        Mat mean, mean_sq;
        blur(src, mean, Size(3,3));
        blur(src.mul(src), mean_sq, Size(3,3));
        variance = mean_sq - mean.mul(mean);
        
        // 将variance转换为float类型
        Mat variance_float;
        variance.convertTo(variance_float, CV_32F);
        
        // 根据局部方差自适应调整锐化强度
        Mat sharpening_strength;
        exp(-variance_float/100.0, sharpening_strength);
        
        // 应用自适应锐化
        src.convertTo(dst, CV_32F);
        dst = dst + laplacian.mul(sharpening_strength);
        dst.convertTo(dst, CV_8U);
    }
};

int main() {
    // 打开视频文件或摄像头
    VideoCapture cap("/home/wys/Desktop/Project/VisionProject/Opencv/images/3.mp4"); // 使用摄像头，或替换为视频文件路径
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }
    
    // 获取视频属性
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    
    // 创建视频写入器
    VideoWriter video("/home/wys/Desktop/Project/VisionProject/Opencv/images/deblurred_output2.avi", 
                     VideoWriter::fourcc('M','J','P','G'),
                     fps,
                     Size(frame_width, frame_height));
    
    ImageDeblur deblurrer;
    Mat frame;
    
    while(1) {
        cap >> frame;
        if (frame.empty())
            break;
            
        Mat processed_frame = deblurrer.processFrame(frame);
        
        imshow("Original", frame);
        imshow("Deblurred", processed_frame);
        
        video.write(processed_frame);
        
        char c = (char)waitKey(1);
        if(c == 'q')
            break;
    }
    
    cap.release();
    video.release();
    destroyAllWindows();
    
    return 0;
}
