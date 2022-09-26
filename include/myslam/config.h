#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{
class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {} // 单例模式，默认构造函数设置为私有
public:
    ~Config();  // close the file when deconstructing

    // 设置一个新的配置文件
    static void setParameterFile(const std::string& filename);

    // 访问参数值
    template<typename T>
    static T get(const std::string& key)
    {
        return T(Config::config_->file_[key]);
    }
};
}

#endif