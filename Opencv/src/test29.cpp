#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

// 定义Rubbiish结构体，存储检测结果
struct Rubbiish {
    cv::Rect boundingBox;
    // 可以添加更多字段来存储其他信息
};

// GreenRubbish类，用于图像处理和物体检测
class GreenRubbish {
public:
    void imgPreProcess(const cv::Mat& frame, cv::Mat& frameGray) {
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);  // 将图像转换为灰度图
    }

    Rubbiish detectObject(const cv::Mat& frameGray, const cv::Mat& frame) {
        // 假设这里是一个简单的物体检测函数，返回一个简单的检测框
        Rubbiish detected;
        detected.boundingBox = cv::Rect(50, 50, 100, 100);  // 示例，假设检测到一个区域
        return detected;
    }
};
// 线程池类
class ThreadPool {
public:
    // 构造函数，指定线程数量
    ThreadPool(size_t numThreads) : stop(false) {
        // 启动指定数量的线程
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    // 提交任务
    template <typename F>
    std::future<void> enqueue(F&& f) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.push([task] { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    // 停止线程池
    void stopAll() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    ~ThreadPool() {
        if (!stop) stopAll();
    }

private:
    std::vector<std::thread> workers;  // 工作线程
    std::queue<std::function<void()>> tasks;  // 任务队列
    std::mutex queueMutex;  // 任务队列的互斥锁
    std::condition_variable condition;  // 条件变量，用于通知空闲线程
    bool stop;  // 是否停止线程池
};
int main() {
    cv::VideoCapture cap(0);  // 打开默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> frames(5);  // 存储多个帧的容器
    std::vector<std::future<void>> futures(5);  // 存储每个任务的future对象
    std::vector<Rubbiish> results(5);  // 存储每帧处理结果的容器
    GreenRubbish greenRubbish;  // 创建物体检测器实例

    // 创建线程池实例，假设我们有4个线程
    ThreadPool threadPool(4);

    while (true) {
        // 从摄像头获取5帧
        for (int i = 0; i < 5; ++i) {
            cap >> frames[i];
            if (frames[i].empty()) {
                std::cerr << "Error: Empty frame!" << std::endl;
                return -1;
            }
        }

        // 提交图像处理任务到线程池
        for (size_t i = 0; i < frames.size(); ++i) {
            futures[i] = threadPool.enqueue([i, &frames, &results, &greenRubbish] {
                cv::Mat frameGray;
                greenRubbish.imgPreProcess(frames[i], frameGray);  // 图像预处理
                Rubbiish result = greenRubbish.detectObject(frameGray, frames[i]);  // 物体检测

                // 存储处理结果
                results[i] = result;
            });
        }

        // 等待所有任务完成
        for (auto& f : futures) {
            f.get();  // 等待每个任务完成
        }

        // 处理完所有帧后，可以对results中的每个结果进行进一步处理
        for (const auto& result : results) {
            std::cout << "Detected Object at: " << result.boundingBox << std::endl;
        }

        // 如果不希望无限循环，可以在某些条件下跳出循环
        // if (some_condition) break;
    }

    return 0;
}
