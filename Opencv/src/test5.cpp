#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

void SobelEdgeDetection(const cv::Mat& src, cv::Mat& edges, bool is_blurred = true) {
    // Convert to grayscale
    cv::Mat gray;
    if(src.channels() > 1){
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = src;
    }
    
    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    if(is_blurred == false)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);
    else 
    blurred = gray;

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

void ScharrEdgeDetection(const cv::Mat& src, cv::Mat& edges, bool is_blurred = true) {
    // Convert to grayscale
    cv::Mat gray;
    if(src.channels() > 1){
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = src;
    }

    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    if(is_blurred == false)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);
    else 
    blurred = gray;
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

void LaplacianEdgeDetection(const cv::Mat& src, cv::Mat& edges, bool is_blurred = true) {
    // Convert to grayscale
    cv::Mat gray;
    if(src.channels() > 1){ 
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = src;
    }
    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    if(is_blurred == false)         
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);
    else                           
    blurred = gray;
    // Apply Laplacian operator
    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_16S, 3);

    // Convert to absolute values
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian);

    // Normalize the result
    cv::normalize(abs_laplacian, edges, 0, 255, cv::NORM_MINMAX, CV_8U);
}


void prewittEdgeDetection(const cv::Mat& src, cv::Mat& edges) {
    // Convert to grayscale if the image is in color
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Define Prewitt kernels
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << 
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 
        1, 1, 1,
        0, 0, 0,
        -1, -1, -1);

    // Apply the kernels to the grayscale image
    cv::Mat grad_x, grad_y;
    cv::filter2D(gray, grad_x, CV_32F, kernel_x);
    cv::filter2D(gray, grad_y, CV_32F, kernel_y);

    // Compute the magnitude of the gradient
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);

    // Convert to absolute values
    cv::convertScaleAbs(magnitude, edges);
}

void RobertsEdgeDetection(const cv::Mat& src, cv::Mat& edges, bool is_blurred = true) {
    // Define Roberts kernels
    cv::Mat robertsX = (cv::Mat_<char>(2, 2) << 1, 0, 0, -1);
    cv::Mat robertsY = (cv::Mat_<char>(2, 2) << 0, 1, -1, 0);

    // Convert to grayscale
    cv::Mat gray;
    if(src.channels() > 1){ 
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = src.clone();
    }
    // Apply GaussianBlur to reduce noise
    cv::Mat blurred;
    if(is_blurred == false)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4, 1.4);
    else
    blurred = gray.clone(); // 这里必须clone，否则后面addWeighted会改变原图

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


// 封装Harris角点检测的函数
// qualityLevel: 0.01-0.001(越小检测角点越多), blockSize: 3-10(越大越平滑，噪声越多), k: 0.04-0.06(加权因子)

// 封装Harris角点检测的函数
void harrisCornerDetection(const cv::Mat& src, cv::Mat& output, double qualityLevel = 0.01, int blockSize = 2, int k = 3) {
    // 将图像转换为灰度图
    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用Harris角点检测
    cv::Mat harris_corners;
    cv::cornerHarris(gray, harris_corners, blockSize, k, qualityLevel);

    // 归一化结果并转换为可显示的格式
    cv::Mat corners;
    cv::normalize(harris_corners, corners, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::convertScaleAbs(corners, corners);

    // 将output设置为源图像的副本
    output = src.clone();

    // 在原图像上绘制角点
    for (int i = 0; i < harris_corners.rows; i++) {
        for (int j = 0; j < harris_corners.cols; j++) {
            // 选择高于阈值的角点
            if ((int)harris_corners.at<float>(i, j) > 200) { // 阈值可以根据需要调整
                cv::circle(output, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, 8, 0); // 在源图上标记角点
            }
        }
    }
}


// 封装Shi-Tomasi角点检测的函数
// maxCorners: 最大角点数, qualityLevel: 0.01-0.001(越小检测角点越多), minDistance: 10-50(两个角点之间的最小距离)

void shiTomasiCornerDetection(const cv::Mat& src, cv::Mat& output, int maxCorners = 100, double qualityLevel = 0.01, double minDistance = 10) {
    // 将图像转换为灰度图
    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 计算角点
    std::vector<cv::Point2f> corners; // 使用std::vector存储角点
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制角点
    for (const auto& corner : corners) {
        cv::circle(output, corner, 5, cv::Scalar(0, 0, 255), -1); // 用红色圆圈表示角点
    }
}


// 封装FAST角点检测的函数
// threshold: 阈值(10-30)越大越强, nonmaxSuppression: 是否进行非极大值抑制(是否去除角点之间的重叠区域)
cv::Mat fastCornerDetection(const cv::Mat& src, int threshold = 10, bool nonmaxSuppression = true) {
    // 将图像转换为灰度图
    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 检测FAST特征点
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, threshold, nonmaxSuppression);

    // 在图像上绘制角点
    cv::Mat output_image = src.clone();
    for (const auto& keypoint : keypoints) {
        cv::circle(output_image, keypoint.pt, 5, cv::Scalar(0, 0, 255), -1); // 用红色圆圈标记角点
    }

    return output_image; // 返回带有角点标记的图像
}


//SIFT特征点检测
// void siftKeypointDetection(const cv::Mat& src, cv::Mat& output) {
//     // 创建SIFT检测器
//     cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
//     // 检测关键点和计算描述符
//     std::vector<cv::KeyPoint> keypoints;
//     cv::Mat descriptors;
//     sift->detectAndCompute(src, cv::noArray(), keypoints, descriptors);
//     // 将输出图像设置为源图像的副本
//     output = src.clone();
//     // 在输出图像上绘制关键点
//     cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// }

//SURF特征点检测
// void surfKeypointDetection(const cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int hessianThreshold = 400) {
//     // 创建SURF检测器
//     cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianThreshold);
//     // 检测关键点和计算描述符
//     surf->detectAndCompute(src, cv::noArray(), keypoints, descriptors);
// }

//ORB特征点检测
void orbKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nfeatures = 500) {
    // 创建ORB检测器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures);

    // 检测关键点和计算描述符
    orb->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// BRIEF特征描述符
void briefKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nfeatures = 500) {
    // 创建FAST检测器来找到特征点
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(nfeatures);

    // 使用FAST检测特征点
    fast->detect(src, keypoints);

    // 创建BRIEF描述子
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();

    // 计算描述符
    brief->compute(src, keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// BRISK特征点检测
void briskKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int thresh = 30, int nOctaves = 4) {
    // 创建BRISK检测器
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(thresh, nOctaves);

    // 检测关键点和计算描述符
    brisk->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// KAZE特征点检测
void kazeKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // 创建KAZE检测器
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();

    // 检测关键点和计算描述符
    kaze->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// AKAZE特征点检测
void akazeKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, float threshold = 0.001f, int nOctaves = 4) {
    // 创建AKAZE检测器
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, threshold, nOctaves);

    // 检测关键点和计算描述符
    akaze->detectAndCompute(src, cv::noArray(), keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

//FREAK特征点检测
void freakKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // 创建FAST检测器并检测特征点
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(src, keypoints);

    // 创建FREAK描述子计算器
    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();

    // 计算描述符
    freak->compute(src, keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

//LATCH特征点检测
void latchKeypointDetection(const cv::Mat& src, cv::Mat& output, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int threshold = 30) {
    // 创建FAST检测器以检测特征点
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(threshold);

    // 检测关键点
    fast->detect(src, keypoints);

    // 创建LATCH描述子计算器
    cv::Ptr<cv::xfeatures2d::LATCH> latch = cv::xfeatures2d::LATCH::create();

    // 计算描述符
    latch->compute(src, keypoints, descriptors);

    // 将输出图像设置为源图像的副本
    output = src.clone();

    // 在输出图像上绘制关键点
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}


int main(){



    // cv::Mat img = cv::imread("/home/wys/Desktop/Project/VisionProject/Opencv/images/origin/1.png",cv::IMREAD_GRAYSCALE);
    // cv::Mat img_color = cv::imread("/home/wys/Desktop/Project/VisionProject/Opencv/images/origin/1.png",cv::IMREAD_COLOR);
    // if(img.empty() || img_color.empty()){
    //     std::cerr << "Could not open or find the image" << std::endl;
    //     return -1;
    // }
    cv::Mat img, img_color;
    cv::VideoCapture cap(2);
    if(!cap.isOpened()){    
        std::cerr << "Could not open the camera" << std::endl;
        return -1;
    }

    
while(cap.isOpened()){
    cap >> img_color;
    cv::cvtColor(img_color, img, cv::COLOR_BGR2GRAY);
    // //Cany边缘监测
    //     cv::Mat canny_output;
    //     cv::Canny(img, canny_output, 50, 150); //lower and upper threshold 控制边缘检测的范围
    // //Sobel边缘监测
    //     cv::Mat sobel_output;
    //     SobelEdgeDetection(img, sobel_output, false); //is_blurred = false 控制是否对图像进行高斯模糊处理
    // //Scharr边缘监测
    //     cv::Mat scharr_output;
    //     ScharrEdgeDetection(img, scharr_output, false); //is_blurred = false 控制是否对图像进行高斯模糊处理
    // //Laplacian边缘监测
    //     cv::Mat laplacian_output;
    //     LaplacianEdgeDetection(img, laplacian_output, false); //is_blurred = false 控制是否对图像进行高斯模糊处理
    // //Prewitt边缘监测
    //     cv::Mat prewitt_output;
    //     prewittEdgeDetection(img_color, prewitt_output);
    // //Roberts边缘监测
    //     cv::Mat roberts_output;
    //     RobertsEdgeDetection(img_color, roberts_output, false); //is_blurred = false 控制是否对图像进行高斯模糊处理
    //Harris角点检测
        cv::Mat harris_output;
        harrisCornerDetection(img, harris_output);
    //Shi-Tomasi角点检测
        cv::Mat shi_tomasi_output;
        shiTomasiCornerDetection(img, shi_tomasi_output);
    //FAST角点检测
        cv::Mat fast_output = fastCornerDetection(img, 10, true);   
    // //SIFT特征点检测
    //     // cv::Mat sift_output;
    //     // siftKeypointDetection(img_color, sift_output);
    // //SURF特征点检测
    //     // std::vector<cv::KeyPoint> surf_keypoints;
    //     // cv::Mat surf_descriptors;
    //     // surfKeypointDetection(img_color, surf_keypoints, surf_descriptors);
    // //ORB特征点检测
    //     cv::Mat orb_output;
    //     std::vector<cv::KeyPoint> orb_keypoints;
    //     cv::Mat orb_descriptors;
    //     orbKeypointDetection(img_color, orb_output, orb_keypoints, orb_descriptors);
    // //BRIEF特征描述符
    //     cv::Mat brief_output;
    //     std::vector<cv::KeyPoint> brief_keypoints;
    //     cv::Mat brief_descriptors;
    //     briefKeypointDetection(img_color, brief_output, brief_keypoints, brief_descriptors);
    // //BRISK特征点检测
    //     cv::Mat brisk_output;
    //     std::vector<cv::KeyPoint> brisk_keypoints;
    //     cv::Mat brisk_descriptors;
    //     briskKeypointDetection(img_color, brisk_output, brisk_keypoints, brisk_descriptors);
    // //KAZE特征点检测
    //     cv::Mat kaze_output;
    //     std::vector<cv::KeyPoint> kaze_keypoints;
    //     cv::Mat kaze_descriptors;
    //     kazeKeypointDetection(img_color, kaze_output, kaze_keypoints, kaze_descriptors);
    // //AKAZE特征点检测
    //     cv::Mat akaze_output;
    //     std::vector<cv::KeyPoint> akaze_keypoints;
    //     cv::Mat akaze_descriptors;
    //     akazeKeypointDetection(img_color, akaze_output, akaze_keypoints, akaze_descriptors);
    // //FREAK特征点检测
    //     cv::Mat freak_output;
    //     std::vector<cv::KeyPoint> freak_keypoints;
    //     cv::Mat freak_descriptors;
    //     freakKeypointDetection(img_color, freak_output, freak_keypoints, freak_descriptors);
    // //LATCH特征点检测
    //     cv::Mat latch_output;
    //     std::vector<cv::KeyPoint> latch_keypoints;
    //     cv::Mat latch_descriptors;
    //     latchKeypointDetection(img_color, latch_output, latch_keypoints, latch_descriptors);    

        // cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Canny Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Sobel Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Scharr Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Laplacian Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Prewitt Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Roberts Output", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Harris Output", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Shi-Tomasi Output", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("FAST Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("SIFT Output", cv::WINDOW_AUTOSIZE);     //
        // cv::namedWindow("SURF Output", cv::WINDOW_AUTOSIZE);    //
        // cv::namedWindow("ORB Output", cv::WINDOW_AUTOSIZE);     
        // cv::namedWindow("BRIEF Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("BRISK Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("KAZE Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("AKAZE Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("FREAK Output", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("LATCH Output", cv::WINDOW_AUTOSIZE);


        // cv::resize(img, img, cv::Size(640, 480));
        cv::resize(img_color, img_color, cv::Size(640, 480));
        // cv::resize(canny_output, canny_output, cv::Size(640, 480));
        // cv::resize(sobel_output, sobel_output, cv::Size(640, 480));
        // cv::resize(scharr_output, scharr_output, cv::Size(640, 480));
        // cv::resize(laplacian_output, laplacian_output, cv::Size(640, 480));
        // cv::resize(prewitt_output, prewitt_output, cv::Size(640, 480));
        // cv::resize(roberts_output, roberts_output, cv::Size(640, 480));
        cv::resize(harris_output, harris_output, cv::Size(640, 480));
        cv::resize(shi_tomasi_output, shi_tomasi_output, cv::Size(640, 480));
        cv::resize(fast_output, fast_output, cv::Size(640, 480));
        // cv::resize(sift_output, sift_output, cv::Size(640, 480));  //
        // cv::resize(surf_output, surf_output, cv::Size(640, 480));  //
        // cv::resize(orb_output, orb_output, cv::Size(640, 480));     
        // cv::resize(brief_output, brief_output, cv::Size(640, 480));
        // cv::resize(brisk_output, brisk_output, cv::Size(640, 480));
        // cv::resize(kaze_output, kaze_output, cv::Size(640, 480));
        // cv::resize(akaze_output, akaze_output, cv::Size(640, 480));
        // cv::resize(freak_output, freak_output, cv::Size(640, 480));
        // cv::resize(latch_output, latch_output, cv::Size(640, 480));


        // cv::imshow("Original Image", img);
        cv::imshow("Color Image", img_color);
        // cv::imshow("Canny Output", canny_output);
        // cv::imshow("Sobel Output", sobel_output);
        // cv::imshow("Scharr Output", scharr_output);
        // cv::imshow("Laplacian Output", laplacian_output);
        // cv::imshow("Prewitt Output", prewitt_output);
        // cv::imshow("Roberts Output", roberts_output);
        cv::imshow("Harris Output", harris_output);
        cv::imshow("Shi-Tomasi Output", shi_tomasi_output);
        cv::imshow("FAST Output", fast_output);
        // cv::imshow("SIFT Output", sift_output);
        // cv::imshow("SURF Output", surf_output);
        // cv::imshow("ORB Output", orb_output);     
        // cv::imshow("BRIEF Output", brief_output);
        // cv::imshow("BRISK Output", brisk_output);
        // cv::imshow("KAZE Output", kaze_output);
        // cv::imshow("AKAZE Output", akaze_output);
        // cv::imshow("FREAK Output", freak_output);
        // cv::imshow("LATCH Output", latch_output);

        cv::waitKey(10);

}

    cv::destroyAllWindows();

    return 0;   
}