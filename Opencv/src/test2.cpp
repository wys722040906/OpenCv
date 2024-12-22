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
    cv::Mat image = cv::imread("/home/wys/Downloads/images/1.jpg");
    if (image.empty()) {
        std::cerr << "Error opening image" << std::endl;
        return -1;
    }
    cv::namedWindow("Origin",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gauss",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Salt",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Poisson",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Union",cv::WINDOW_AUTOSIZE);
    
    cv::Mat Gaussian_noisyImage = addGaussianNoise(image, 0, 30);
    cv::Mat Salt_noisyImahe = addSaltAndPepperNoise(image, 0.2);
    cv::Mat PoissonNoise = addPoissonNoise(image);
    cv::Mat UniNoise = addUniformNoise(image, 50);
    
    cv::resize(image,image,cv::Size(640,480));
    cv::resize(Gaussian_noisyImage,Gaussian_noisyImage,cv::Size(640,480));
    cv::resize(Salt_noisyImahe,Salt_noisyImahe,cv::Size(640,480));
    cv::resize(PoissonNoise,PoissonNoise,cv::Size(640,480));
    cv::resize(UniNoise,UniNoise,cv::Size(640,480));
    
    cv::imshow("Origin", image);
    cv::imshow("Gauss", Gaussian_noisyImage);
    cv::imshow("Salt", Salt_noisyImahe);
    cv::imshow("Poisson", PoissonNoise);
    cv::imshow("Union", UniNoise);
    cv::waitKey(0);
    return 0;
}
