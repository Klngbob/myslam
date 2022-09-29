#include <opencv2/calib3d/calib3d.hpp>

#include "myslam/visual_odometry.h"
#include "myslam/config.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry():
    state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map),
    num_lost_(0), num_inliers_(0)
{
    num_of_features_    = Config::get<int>("number_of_features");
    scale_factor_       = Config::get<double>("scale_factor");
    level_pyramid_      = Config::get<int>("level_pyramid");
    match_ratio_        = Config::get<float>("match_ratio");
    max_num_lost_       = Config::get<int>("max_num_lost");
    min_inliers_        = Config::get<int>("min_inliers");
    key_frame_min_rot_  = Config::get<double>("key_frame_min_rot");
    key_frame_min_trans_= Config::get<double>("key_frame_min_trans");
    orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame(Frame::Ptr frame)
{
    switch (state_)
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        map_->insertKeyFrame(frame);
        // 从第一帧里提取特征
        extractKeyPoints();
        computeDescriptors();
        // 计算参考帧中特征点的3D位置
        setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if(checkEstimatedPose() == true) // 一个好的估计
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;
            ref_ = curr_;
            setRef3DPoints();
            num_lost_ = 0;
            if(checkKeyFrame() == true) // 如果是一个关键帧
            {
                addKeyFrame();
            }
        }
        else 
        {
            num_lost_++;
            if(num_lost_ > max_num_lost_)
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"VO has lost."<<endl;
        break;
    }
    }
    return true;
}

// 提取特征点
void VisualOdometry::extractKeyPoints()
{
    orb_->detect(curr_->color_, keypoints_curr_);
}

// 计算描述子
void VisualOdometry::computeDescriptors()
{
    orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
}

// 特征匹配
void VisualOdometry::featureMatching()
{
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_ref_, descriptors_curr_, matches);
    float min_dis = std::min_element(
        matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;}
    )->distance;

    // 根据距离筛选特征点
    feature_matches_.clear();
    for(cv::DMatch& m : matches)
    {
        if(m.distance < max<float>(min_dis * match_ratio_, 30.0))
            feature_matches_.push_back(m);
    }
    cout<<"good matches: "<<feature_matches_.size()<<endl;
}

// pnp需要参考帧3D，当前帧2D，所以当前帧迭代为参考帧时，需要加上深度数据
void VisualOdometry::setRef3DPoints()
{
    // 选择带有深度测量值的特征
    pts_3d_ref_.clear();
    descriptors_ref_ = Mat();
    for(size_t i = 0; i < keypoints_curr_.size(); ++i)
    {
        // 找到depth数据
        double d = ref_->findDepth(keypoints_curr_[i]);
        if(d > 0)
        {
            // 像素坐标到相机坐标系3D坐标
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
                d
            );
            pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
            // 参考帧描述子，将当前帧描述子按行放进去
            descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
}

// PnP估计相机位姿
void VisualOdometry::poseEstimationPnP()
{
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    // 参考帧3D坐标和当前帧2D坐标
    for(cv::DMatch m : feature_matches_)
    {
        pts3d.push_back(pts_3d_ref_[m.queryIdx]);
        pts2d.push_back(keypoints_curr_[m.trainIdx].pt);
    }
    // 相机内参
    Mat K = (cv::Mat_<double>(3, 3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0,ref_->camera_->fy_, ref_->camera_->cy_,
        0, 0, 1);
    // 旋转、平移、内点数组
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    // 内点数为内点行数
    num_inliers_ = inliers.rows; // ?
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    // 由旋转和平移构造出的当前帧相对于参考帧的变换矩阵T
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
        Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
    );

    // 使用bundle adjustment优化估计的位姿T
    // 1.位姿6维，观测点2维
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // 2.顶点是相机位姿pose
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
        T_c_r_estimated_.rotation_matrix(),
        T_c_r_estimated_.translation()
    ));
    optimizer.addVertex(pose);

    // 3.边是重投影误差
    for(int i = 0; i < inliers.rows; ++i)
    {
        int index = inliers.at<int>(i, 0);
        // 3D->2D投影
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        // 相机参数
        edge->camera_ = curr_->camera_.get();
        // 3D点
        edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
        // 测量值是2维
        edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    // 4.执行优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    T_c_r_estimated_ = SE3(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
}

// 位姿检验模块，匹配点不能太少，运动不能太大
bool VisualOdometry::checkEstimatedPose()
{
    // 匹配点太少
    if(num_inliers_ < min_inliers_)
    {
        cout<<"Reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // T的模太大
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if(d.norm() > 5.0)
    {
        cout<<"Reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

// 检查关键帧，旋转或平移的模大于给定的最小值即可
bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if(rot.norm() > key_frame_min_rot_ || trans.norm() > key_frame_min_trans_)
        return true;
    return false;
}

// 添加关键帧
void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame(curr_);
}
}