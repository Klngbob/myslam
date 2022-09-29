#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam
{

class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned        long id_;  // id
    static unsigned long factory_id_; // 工厂模式的id
    bool            good_;    // 是否是一个好的点
    Vector3d        pos_;     // 世界坐标
    Vector3d        norm_;    // Normal of viewing direction(法线)
    Mat             descriptor_;   // 描述子

    list<Frame*>    observed_frames_;    // 能观测到这个点的关键帧

    // int             observed_times_;    // 一个点被特征匹配观测到的次数
    // int             correct_times_;     // 被匹配的次数
    int             matched_times_; // being an inliner in pose estimation
    int             visible_times_; // being visible in current frame    

    MapPoint();
    MapPoint(
        unsigned long id,
        const Vector3d& position,
        const Vector3d& norm,
        Frame* frame = nullptr,
        const Mat& descriptor=Mat()
    );

    inline cv::Point3f getPositionCV() const {
        return cv::Point3f(pos_(0, 0), pos_(1 ,0), pos_(2, 0));
    }

    // 工厂模式
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint(
        const Vector3d& pos_world,
        const Vector3d& norm,
        const Mat& descriptor,
        Frame* frame
    );
};

}

#endif