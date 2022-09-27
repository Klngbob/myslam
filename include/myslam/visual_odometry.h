#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/frame.h"
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };
    VOState     state_;  // 当前VO状态
    Map::Ptr    map_;   // 所有帧和所有地图点构成的地图
    Frame::Ptr  ref_;  // 参考帧
    Frame::Ptr  curr_; // 当前帧

    cv::Ptr<cv::ORB> orb_; // ORB detector and computer
    vector<cv::Point3f> pts_3d_ref_; // 参考帧中的3d点
    vector<cv::KeyPoint> keypoints_curr_; // 当前帧中的关键点
    Mat descriptors_curr_; // 当前帧中的描述子
    Mat descriptors_ref_;  // 参考帧中的描述子
    vector<cv::DMatch> feature_matches_; // 描述子匹配关系

    SE3 T_c_r_estimated_; // 当前帧的估计位姿
    int num_inliers_;    // icp中内点数
    int num_lost_;       // 丢失次数

    // 参数
    int num_of_features_; // 特征数
    double scale_factor_; // scale in image pyramid
    int level_pyramid_;   // 金字塔层数
    float match_ratio_;   // 选择好的匹配的比率
    int max_num_lost_;    // 最大连续丢失次数
    int min_inliers_;     // 最小内点数

    double key_frame_min_rot_; // 两个关键帧的最小旋转
    double key_frame_min_trans_; // 两个关键帧的最小平移
public:
    VisualOdometry();
    ~VisualOdometry();

    bool addFrame(Frame::Ptr frame); // 添加一个新帧

protected:
    // 内部处理函数
    void extractKeyPoints(); // 提取关键点
    void computeDescriptors(); // 计算描述子
    void featureMatching();    // 特征点匹配并筛选
    void poseEstimationPnP();
    void setRef3DPoints();
    // 关键帧的功能函数
    void addKeyFrame();
    bool checkEstimatedPose();
    bool checkKeyFrame();
};
}

#endif