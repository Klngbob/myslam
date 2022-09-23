#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/mappoint.h"
#include "myslam/frame.h"

namespace myslam
{

class Map
{
public:
    typedef std::unique_ptr<Map> Ptr;
    unordered_map<unsigned long, MapPoint::Ptr> map_points_; // 所有路标关键点
    unordered_map<unsigned long, Frame::Ptr> keyframes_;     // 所有关键帧

    Map() {}
    
    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);
};

} // namespace myslam


#endif // MAP_H