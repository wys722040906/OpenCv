#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

class VideoRecorder {
private:
    VideoCapture cap;
    VideoWriter writer;
    bool is_recording;
    string output_path;
    const int WIDTH = 640;
    const int HEIGHT = 480;
    const double FPS = 30.0;

    struct ThreshParams {
        double low_H = 0;
        double high_H = 180;
        double low_S = 30;
        double high_S = 255;
        double low_V = 30;
        double high_V = 255;
    } thresh_params;

    // 篮球检测相关
    struct AdaptiveParams {
        double mean_v = 0;
        double std_v = 0;
        double mean_s = 0;
        bool is_initialized = false;
    } adaptive_params;

    struct DetectionResult {
        bool valid;
        vector<Point> contour;
        Point2f center;
        float radius;
    };

    Mat frame, hsv, mask;
    bool detection_mode = false;  // 新增：检测模式开关

    void updateAdaptiveParams(const Mat& hsv) {
        vector<Mat> channels;
        split(hsv, channels);
        
        Mat s = channels[1];
        Mat v = channels[2];
        
        Scalar mean_s, std_s, mean_v, std_v;
        meanStdDev(s, mean_s, std_s);
        meanStdDev(v, mean_v, std_v);
        
        if (!adaptive_params.is_initialized) {
            adaptive_params.mean_v = mean_v[0];
            adaptive_params.std_v = std_v[0];
            adaptive_params.mean_s = mean_s[0];
            adaptive_params.is_initialized = true;
        } else {
            const double alpha = 0.1;
            adaptive_params.mean_v = (1-alpha) * adaptive_params.mean_v + alpha * mean_v[0];
            adaptive_params.std_v = (1-alpha) * adaptive_params.std_v + alpha * std_v[0];
            adaptive_params.mean_s = (1-alpha) * adaptive_params.mean_s + alpha * mean_s[0];
        }
        
        adjustThresholds();
    }

    void adjustThresholds() {
        thresh_params.low_V = max(30.0, adaptive_params.mean_v - 2 * adaptive_params.std_v);
        thresh_params.high_V = min(255.0, adaptive_params.mean_v + 2 * adaptive_params.std_v);
        thresh_params.low_S = max(30.0, adaptive_params.mean_s - adaptive_params.std_v);
    }

    Mat preprocess(const Mat& input) {
        Mat processed = input.clone();
        
        // 1. 白平衡校正
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        Mat lab;
        cvtColor(processed, lab, COLOR_BGR2Lab);
        vector<Mat> lab_channels;
        split(lab, lab_channels);
        clahe->apply(lab_channels[0], lab_channels[0]);
        merge(lab_channels, lab);
        cvtColor(lab, processed, COLOR_Lab2BGR);
        
        // 2. 转换到HSV空间
        cvtColor(processed, hsv, COLOR_BGR2HSV);
        
        // 3. 更新自适应参数
        updateAdaptiveParams(hsv);
        
        // 4. 创建掩码
        mask = createRobustMask();
        
        // 5. 形态学处理
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        
        return processed;
    }

    Mat createRobustMask() {
        Mat mask1, mask2, combined_mask;
        
        // YCrCb空间检测
        Mat ycrcb;
        cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
        vector<Mat> ycrcb_channels;
        split(ycrcb, ycrcb_channels);
        
        // HSV检测
        inRange(hsv, 
                Scalar(thresh_params.low_H, thresh_params.low_S, thresh_params.low_V),
                Scalar(thresh_params.high_H, thresh_params.high_S, thresh_params.high_V),
                mask1);
        
        // YCrCb辅助检测
        Mat cr_mask;
        inRange(ycrcb_channels[1], 140, 180, cr_mask);
        
        bitwise_and(mask1, cr_mask, combined_mask);
        
        return combined_mask;
    }

    DetectionResult detectBasketball(const Mat& processed) {
        DetectionResult result;
        result.valid = false;
        
        vector<vector<Point>> contours;
        findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // 找到最大的轮廓
        if (!contours.empty()) {
            auto max_contour = max_element(contours.begin(), contours.end(),
                [](const vector<Point>& c1, const vector<Point>& c2) {
                    return contourArea(c1) < contourArea(c2);
                });
                
            // 计算最小包围圆
            Point2f center;
            float radius;
            minEnclosingCircle(*max_contour, center, radius);
            
            // 验证检测结果
            if (validateDetection(*max_contour, radius)) {
                result.valid = true;
                result.contour = *max_contour;
                result.center = center;
                result.radius = radius;
            }
        }
        
        return result;
    }

    bool validateDetection(const vector<Point>& contour, float radius) {
        // 面积检查
        double area = contourArea(contour);
        if (area < 1000 || area > 50000) return false;
        
        // 圆度检查
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.6) return false;
        
        return true;
    }

public:
    VideoRecorder(int camera_index) : is_recording(false) {
        // 使用整数索引打开摄像头
        cap.open(camera_index, cv::CAP_V4L2);  // Linux系统使用V4L2后端
        if (!cap.isOpened()) {
            cout << "V4L2后端失败，尝试默认后端..." << endl;
            cap.open(camera_index);  // 尝试默认后端
        }
        
        if (!cap.isOpened()) {
            throw runtime_error("无法打开摄像头 " + to_string(camera_index));
        }

        // 设置摄像头参数
        cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        cap.set(CAP_PROP_FPS, FPS);
        
        cout << "成功打开摄像头 " << camera_index << endl;
    }

    void run() {
        namedWindow("Camera", WINDOW_AUTOSIZE);
        namedWindow("Processed", WINDOW_AUTOSIZE);
        // createTrackbars();  // 原有的阈值控制

        cout << "\n控制说明：" << endl;
        cout << "'r': 开始/停止录制" << endl;
        cout << "'s': 保存当前帧" << endl;
        cout << "'t': 切换阈值调整模式" << endl;
        cout << "'d': 切换篮球检测模式" << endl;
        cout << "'q': 退出程序\n" << endl;

        while (true) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            Mat display_frame = frame.clone();
            Mat processed_frame;
            DetectionResult detection;

            if (detection_mode) {
                processed_frame = preprocess(frame);
                detection = detectBasketball(processed_frame);
                
                if (detection.valid) {
                    // 绘制检测结果
                    circle(display_frame, detection.center, detection.radius, 
                           Scalar(0, 255, 0), 2);
                    drawContours(display_frame, vector<vector<Point>>{detection.contour}, 
                               0, Scalar(0, 0, 255), 2);
                }
                
                imshow("Processed", mask);  // 显示处理后的掩码
            }

            // 显示状态信息
            string status = is_recording ? "Recording..." : "Press 'r' to record";
            status += detection_mode ? " | Detection Mode" : "";
            putText(display_frame, status, Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 0.7,
                   is_recording ? Scalar(0, 0, 255) : Scalar(255, 255, 255),
                   2);

            imshow("Camera", display_frame);

            // 处理按键
            char key = waitKey(10);
            if (key == 'q' || key == 'Q') {
                break;
            }
            else if (key == 'd' || key == 'D') {
                detection_mode = !detection_mode;
                cout << (detection_mode ? "进入篮球检测模式" : "退出篮球检测模式") << endl;
            }
            else if (key == 'r' || key == 'R') {
                if (!is_recording) {
                    // 开��录制
                    time_t now = time(0);
                    string timestamp = to_string(now);
                    output_path = "output_" + timestamp + ".mp4";
                    
                    writer.open(output_path, 
                              VideoWriter::fourcc('m', 'p', '4', 'v'),
                              FPS, Size(WIDTH, HEIGHT), true);
                              
                    if (!writer.isOpened()) {
                        cout << "无法创建视频文件" << endl;
                        continue;
                    }
                    is_recording = true;
                    cout << "开始录制..." << endl;
                }
                else {
                    // 停止录制
                    writer.release();
                    is_recording = false;
                    cout << "录制完成，视频已保存: " << output_path << endl;
                }
            }
        }
    }

    ~VideoRecorder() {
        if (writer.isOpened()) {
            writer.release();
        }
        cap.release();
        destroyAllWindows();
    }
};

int main() {
    try {
        // 使用整数索引 2
        VideoRecorder recorder(2);  // 直接使用整数 2
        recorder.run();
    }
    catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return -1;
    }
    return 0;
}

