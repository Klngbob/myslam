#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{

class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned        long id_;  // id
    Vector3d        pos_;     // 世界坐标
    Vector3d        norm_;    // Normal of viewing direction
    Mat             descriptor_;   // 描述子
    int             observed_times_;    // 一个点被特征匹配观测到的次数
    int             correct_times_;     // 被匹配的次数

    MapPoint();
    MapPoint(long id, Vector3d position, Vector3d norm);

    // 工厂模式
    static MapPoint::Ptr createMapPoint();
};

}

#endif