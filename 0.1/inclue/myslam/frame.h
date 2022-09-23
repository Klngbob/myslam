#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam 
{
class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long id_;  // 这个帧的id
    double time_stamp_; // 帧的时间戳
    SE3 T_c_w_;  // 世界坐标系到相机坐标系的变换矩阵李群
    Camera::Ptr camera_;  // 针孔RGB_D相机模型
    Mat color_, depth_;  // color and depth image

public:
    Frame();
    Frame(long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr,
        Mat color=Mat(), Mat depth=Mat());
    ~Frame();

    // 工厂模式
    static Frame::Ptr createFrame();

    // 在深度地图中找深度z
    double findDepth(const cv::KeyPoint& kp);

    // 获取相机光心
    Vector3d getCamCenter() const;

    // 判断某个世界坐标系下的点是否在视野内
    bool isInFrame(const Vector3d& pt_world);
};

}

#endif