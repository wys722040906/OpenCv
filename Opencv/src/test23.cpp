#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <map>


// 定义参数写入类
// 该类可以将参数写入yaml文件中，并支持多种数据类型写入
//        单一参数(Mat int double bool string)
//        序列(vector)
//        映射(map)
// 写入模式:
//        WRITE:覆盖写入
//        APPEND:追加写入
//        WRITE_BASE64:base64编码的图像数据--需要先将图像数据编码为base64编码格式，然后写入
//         
class ParameterWriter {
public:
    enum FileMode { WRITE, APPEND, WRITE_BASE64 };

    // 构造函数，添加一个用于创建文件的布尔参数
    ParameterWriter(const std::string& filename, FileMode mode = WRITE, bool createIfNotExists = false) : fs() {
        // 处理 WRITE、APPEND 和 WRITE_BASE64 模式
        int cvMode = (mode == WRITE) ? cv::FileStorage::WRITE :
                     (mode == APPEND) ? cv::FileStorage::APPEND :
                     (mode == WRITE_BASE64) ? cv::FileStorage::WRITE_BASE64 : -1;

        // 打开文件
        if (cvMode != -1) {
            fs.open(filename, cvMode);
            if (!fs.isOpened()) {
                if (createIfNotExists) {
                    std::cerr << "File " << filename << " does not exist. Creating the file." << std::endl;
                    fs.open(filename, cv::FileStorage::WRITE);  // 尝试创建新文件
                } else {
                    std::cerr << "Error: Cannot open file " << filename << std::endl;
                }
            }
        }
    }

    // 析构函数
    ~ParameterWriter() {
        if (fs.isOpened()) {
            fs.release();
        }
    }

    // 写入单一参数
    template <typename T>
    void write(const std::string& key, const T& data) {
        if (fs.isOpened()) {
            fs << key << data;
        } else {
            std::cerr << "Error: File not open." << std::endl;
        }
    }

    // 写入序列
    template <typename T>
    void writeSequence(const std::string& key, const std::vector<T>& data) {
        if (fs.isOpened()) {
            fs << key << "[";
            for (const auto& item : data) {
                fs << item;
            }
            fs << "]";
        } else {
            std::cerr << "Error: File not open." << std::endl;
        }
    }

    // 写入映射
    template <typename K, typename V>
    void writeMap(const std::string& key, const std::map<K, V>& data) {
        if (fs.isOpened()) {
            fs << key << "{";
            for (const auto& [mapKey, mapValue] : data) {
                fs << mapKey << mapValue;
            }
            fs << "}";
        } else {
            std::cerr << "Error: File not open." << std::endl;
        }
    }

private:
    cv::FileStorage fs;
};


class ParameterReader {
public:
    // 构造函数
    ParameterReader(const std::string& filename) : fs() {
        // 打开文件
        fs.open(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
        }
    }

    // 析构函数
    ~ParameterReader() {
        if (fs.isOpened()) {
            fs.release();
        }
    }

    // 读取单一参数
    template <typename T>
    bool read(const std::string& key, T& data) {
        if (fs.isOpened()) {
            cv::FileNode node = fs[key];
            if (!node.empty()) {
                node >> data;
                return true;
            } else {
                std::cerr << "Error: Key \"" << key << "\" not found." << std::endl;
                return false;
            }
        } else {
            std::cerr << "Error: File not open." << std::endl;
            return false;
        }
    }

    // 读取序列
    template <typename T>
    bool readSequence(const std::string& key, std::vector<T>& data) {
        if (fs.isOpened()) {
            cv::FileNode node = fs[key];
            if (node.type() == cv::FileNode::SEQ) {
                data.clear(); // 清空向量以便读取
                for (const auto& item : node) {
                    T value;
                    item >> value;
                    data.push_back(value);
                }
                return true;
            } else {
                std::cerr << "Error: Node \"" << key << "\" is not a sequence." << std::endl;
                return false;
            }
        } else {
            std::cerr << "Error: File not open." << std::endl;
            return false;
        }
    }

// 读取映射
template <typename K, typename V>
bool readMap(const std::string& key, std::map<K, V>& data) {
    if (fs.isOpened()) {
        cv::FileNode node = fs[key];
        if (node.type() == cv::FileNode::MAP) {
            data.clear(); // 清空映射以便读取
            for (const auto& item : node) {
                K mapKey;
                V mapValue;

                // 使用item.name()来获取键，item.value()来获取值
                mapKey = item.name();   // 读取键
                item >> mapValue;       // 读取值
                data[mapKey] = mapValue;
            }
            return true;
        } else {
            std::cerr << "Error: Node \"" << key << "\" is not a map." << std::endl;
            return false;
        }
    } else {
        std::cerr << "Error: File not open." << std::endl;
        return false;
    }
}


private:
    cv::FileStorage fs;
};


int main() {
    {
    ParameterWriter writer("/home/wys/Desktop/Project/VisionProject/Opencv/config/config.yaml", ParameterWriter::WRITE, true);

    // 写入参数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    writer.write("camera_matrix", cameraMatrix);

    std::vector<double> focalLengths = {800.0, 850.0, 900.0};
    writer.writeSequence("focal_lengths", focalLengths);

    std::map<std::string, int> imageSize = {{"width", 1280}, {"height", 720}};
    writer.writeMap("image_size", imageSize);

    float a = 1.0f;
    writer.write("a", a);
    }
    {
        ParameterReader reader("/home/wys/Desktop/Project/VisionProject/Opencv/config/config.yaml");
// 读取参数
        cv::Mat cameraMatrix;
        reader.read("camera_matrix", cameraMatrix);
        std::cout << "camera_matrix: " << cameraMatrix << std::endl;
        std::vector<double> focalLengths;
        reader.readSequence("focal_lengths", focalLengths);
        std::cout << "focal_lengths: ";
        for (const auto& item : focalLengths) {
            std::cout << item << " ";
        }
        std::cout << std::endl;

        std::map<std::string, int> imageSize;
        reader.readMap("image_size", imageSize);
        std::cout << "image_size: ";
        for (const auto& [key, value] : imageSize) {
            std::cout << key << ": " << value << " ";
        }
        std::cout << std::endl;
    }



    return 0;
}
