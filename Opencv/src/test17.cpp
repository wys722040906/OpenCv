#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/*单类别的滑动窗口物体检测
    

1.数据集准备：
    正样本：包含目标的与滑动窗口大小相同的图片
    负样本：不包含目标的任意大小图片

    import cv2
    import os
    import glob

    # 创建保存正样本和负样本的目录
    os.makedirs('positives', exist_ok=True)
    os.makedirs('negatives', exist_ok=True)

    # 设置正样本的图像文件夹和对应的 YOLO 格式的标注文件夹
    image_folder = 'path/to/your/images'  # 正样本图像文件夹
    annotations_folder = 'path/to/your/annotations'  # YOLO 格式标注文件夹
    negative_folder = 'path/to/your/negative/images'  # 负样本图像文件夹

    # 提取正样本
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # 获取所有 jpg 图像

    for img_path in image_paths:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        # 生成对应的标注文件名
        annotation_path = os.path.join(annotations_folder, os.path.basename(img_path).replace('.jpg', '.txt'))
        
        # 读取标注文件
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as file:
                for line in file.readlines():
                    # 每一行的格式为: class_id center_x center_y width height (YOLO格式)
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # 获取 YOLO 格式的坐标
                        center_x = float(parts[1]) * width
                        center_y = float(parts[2]) * height
                        obj_width = float(parts[3]) * width
                        obj_height = float(parts[4]) * height

                        # 计算左上角坐标
                        x = int(center_x - obj_width / 2)
                        y = int(center_y - obj_height / 2)

                        # 裁剪正样本
                        positive_sample = img[y:y + int(obj_height), x:x + int(obj_width)]
                        positive_sample = cv2.resize(positive_sample, (64, 128))  # 调整大小
                        cv2.imwrite(f'positives/sample_{os.path.basename(img_path).replace(".jpg", "")}_{len(os.listdir("positives"))}.jpg', positive_sample)

    # 提取负样本
    negative_image_paths = glob.glob(os.path.join(negative_folder, '*.jpg'))  # 获取所有负样本图像

    for img_path in negative_image_paths:
        img = cv2.imread(img_path)
        # 保存负样本到指定目录
        cv2.imwrite(f'negatives/neg_sample_{len(os.listdir("negatives"))}.jpg', img)


2.HOG 特征提取与 SVM 模型训练
    2.1 HOG 特征提取(提取正负样本的Hog特征)：
        import cv2
        from sklearn.svm import SVC
        import os
        import glob

        # HOG特征提取器
        hog = cv2.HOGDescriptor()

        # 准备正负样本数据
        X = []
        y = []

        # 加载正样本
        positive_samples = glob.glob('positives/*.jpg')
        for sample in positive_samples:
            img = cv2.imread(sample)
            features = hog.compute(img).flatten()
            X.append(features)
            y.append(1)  # 正样本标记为1

        # 加载负样本
        negative_samples = glob.glob('negatives/*.jpg')
        for sample in negative_samples:
            img = cv2.imread(sample)
            features = hog.compute(img).flatten()
            X.append(features)
            y.append(0)  # 负样本标记为0

        # 转换为 numpy 数组
        X = np.array(X)
        y = np.array(y)
    2.2 训练 SVM 模型
        from sklearn.model_selection import train_test_split
        import joblib

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练 SVM
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X_train, y_train)

        # 保存模型
        joblib.dump(svm, 'svm_model.pkl')
3.加载模型并进行预测
        # 加载训练好的 SVM 模型
        svm = joblib.load('svm_model.pkl')
        # 实时检测
        cap = cv2.VideoCapture(0)  # 使用摄像头，或使用 cv2.VideoCapture('video.mp4') 来读取视频文件

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # HOG滑动窗口检测
            detections = []
            h, w = frame.shape[:2]
            
            for y in range(0, h - 128, 8):  # 垂直滑动
                for x in range(0, w - 64, 8):  # 水平滑动
                    window = frame[y:y+128, x:x+64]
                    if window.shape[0] != 128 or window.shape[1] != 64:
                        continue  # 确保窗口大小一致
                    
                    # 提取HOG特征
                    features = hog.compute(window).flatten()
                    features = features.reshape(1, -1)  # 适应SVM的输入格式

                    # 进行预测
                    if svm.predict(features) == 1:  # 如果预测为正样本
                        detections.append((x, y, 64, 128))  # 记录窗口位置

            # 在检测到的物体上画框
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
*/

using namespace cv;
using namespace std;
using namespace cv::ml;

int main() {
    // 加载 SVM 模型
    Ptr<SVM> svm = SVM::load("svm_model.yml"); // 假设您将模型保存为 'svm_model.yml'

    // 初始化 HOG 描述符
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // 实时检测
    VideoCapture cap(0); // 使用摄像头，或使用 VideoCapture("video.mp4") 读取视频文件

    if (!cap.isOpened()) {
        cerr << "无法打开摄像头!" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame; // 读取每一帧

        if (frame.empty()) {
            break;
        }

        // HOG滑动窗口检测
        vector<Rect> detections;
        int h = 128; // 窗口高度
        int w = 64;  // 窗口宽度

        for (int y = 0; y <= frame.rows - h; y += 8) { // 垂直滑动
            for (int x = 0; x <= frame.cols - w; x += 8) { // 水平滑动
                Mat window = frame(Rect(x, y, w, h));

                // 提取 HOG 特征
                vector<float> hogFeatures;
                hog.compute(window, hogFeatures);

                // 将特征转换为 Mat 格式以便于 SVM 预测
                Mat featuresMat(hogFeatures).reshape(1, 1); // 转换为一行
                float prediction = svm->predict(featuresMat);

                // 进行预测
                if (prediction == 1) { // 如果预测为正样本
                    detections.push_back(Rect(x, y, w, h)); // 记录窗口位置
                }
            }
        }

        // 在检测到的物体上画框
        for (const auto& detection : detections) {
            rectangle(frame, detection, Scalar(0, 255, 0), 2);
        }

        imshow("Object Detection", frame);

        if (waitKey(1) == 'q') { // 按 'q' 键退出
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
