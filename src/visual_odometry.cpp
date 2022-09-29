#include <opencv2/calib3d/calib3d.hpp>
#include <boost/timer.hpp>

#include "myslam/visual_odometry.h"
#include "myslam/config.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry():
    state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map),
    num_lost_(0), num_inliers_(0), matcher_flann_(new cv::flann::LshIndexParams(5,10,2))
{
    num_of_features_    = Config::get<int>("number_of_features");
    scale_factor_       = Config::get<double>("scale_factor");
    level_pyramid_      = Config::get<int>("level_pyramid");
    match_ratio_        = Config::get<float>("match_ratio");
    max_num_lost_       = Config::get<int>("max_num_lost");
    min_inliers_        = Config::get<int>("min_inliers");
    key_frame_min_rot_  = Config::get<double>("key_frame_min_rot");
    key_frame_min_trans_= Config::get<double>("key_frame_min_trans");
    map_point_erase_ratio_ = Config::get<double>("map_point_erase_ratio");
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
        // map_->insertKeyFrame(frame);
        // 从第一帧里提取特征
        extractKeyPoints();
        computeDescriptors();
        addKeyFrame(); // 第一帧是一个关键帧
        // 计算参考帧中特征点的3D位置
        // setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if(checkEstimatedPose() == true) // 一个好的估计
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if(checkKeyFrame() == true) // 如果是一个关键帧
            {
                addKeyFrame();
            }
        }
        else // 差的估计
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
    boost::timer timer;
    orb_->detect(curr_->color_, keypoints_curr_);
    cout<<"Extract keypoints cost time: "<<timer.elapsed()<<endl;
}

// 计算描述子
void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
    cout<<"Compute descriptors cost time: "<<timer.elapsed()<<endl;
}

// 特征匹配
void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;

    // 建立一个保存描述子的map矩阵，保存匹配需要地图点的描述子(行向量)
    Mat desp_map;
    // 暂时存放在当前帧视野以内的mappoint
    vector<MapPoint::Ptr> candidate;
    // 遍历所有地图点，将符合条件的mappoint放入candidate，将描述子信息放入desp_map
    for(auto& allpoints: map_->map_points_)
    {
        MapPoint::Ptr& p = allpoints.second;
        // 检测这个点（世界坐标系）是否在当前帧视野内
        if(curr_->isInFrame(p->pos_))
        {
            // add to candidate
            p->visible_times_++;
            candidate.push_back(p);
            // Mat 也可以push_back
            desp_map.push_back(p->descriptor_);
        }
    }

    // 使用新的匹配方法FlannBasedMatcher(最近邻近似匹配)，当前帧和地图直接进行匹配
    matcher_flann_.match(desp_map, descriptors_curr_, matches);

    float min_dis = std::min_element(
        matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;}
    )->distance;

    // 
    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for(cv::DMatch& m : matches)
    {
        if(m.distance < max<float>(min_dis * match_ratio_, 30.0))
        {
            match_3dpts_.push_back(candidate[m.queryIdx]);
            match_2dkp_index_.push_back(m.trainIdx);
        }
    }
    cout<<"good matches: "<<match_3dpts_.size()<<endl;
    cout<<"match cost time: "<<timer.elapsed()<<endl;
}

// PnP估计相机位姿
void VisualOdometry::poseEstimationPnP()
{
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    // 去除当前帧匹配成功的2D点
    for(int index: match_2dkp_index_)
    {
        // 将当前帧中的关键点放入pts2d中
        pts2d.push_back(keypoints_curr_[index].pt);
    }
    // 
    for(MapPoint::Ptr pt : match_3dpts_)
    {
        // 匹配好的3D点放进去
        pts3d.push_back(pt->getPositionCV());
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
    // 输出符合模型的数据个数
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    // 由旋转和平移构造出的当前帧相对于参考帧的变换矩阵T
    T_c_w_estimated_ = SE3(
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
        T_c_w_estimated_.rotation_matrix(),
        T_c_w_estimated_.translation()
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

    T_c_w_estimated_ = SE3(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );

    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
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
    // 上一帧到当前帧的变换矩阵T
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();

    Sophus::Vector6d d = T_r_c.log();
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
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if(rot.norm() > key_frame_min_rot_ || trans.norm() > key_frame_min_trans_)
        return true;
    return false;
}

// 增加关键帧
void VisualOdometry::addKeyFrame()
{
    // 第一帧肯定是空的，在提取第一帧的特征点后，将其都放入地图中
    if(map_->keyframes_.empty())
    {
        for(size_t i = 0; i < keypoints_curr_.size(); ++i)
        {
            double d = curr_->findDepth(keypoints_curr_[i]);
            if(d < 0)
                continue;
            Vector3d p_world = ref_->camera_->pixel2world(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
                curr_->T_c_w_,
                d
            );
            // 参考帧的相机中心到当前关键点对应的三维点的位移
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize(); // 变成单位向量
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i), curr_.get()
            );
            map_->insertMapPoint(map_point);
        }
    }
    map_->insertKeyFrame(curr_);
    ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    vector<bool> matched(keypoints_curr_.size(), false);
    for(int index: match_2dkp_index_)
        matched[index] = true;
    for(int i = 0; i < keypoints_curr_.size(); ++i)
    {
        // 如果为真，说明地图中已经有这个匹配的关键点了
        if(matched[i])
            continue;
        
        // 如果为假， 则说明这个点在地图中没有找到匹配，认为是新的点
        // 此时应该找到depth数据，构造3D点，并添加进地图中
        double d = curr_->findDepth(keypoints_curr_[i]);
        if(d < 0)
            continue;
        Vector3d p_world = ref_->camera_->pixel2world(
            Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
            curr_->T_c_w_,
            d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint(map_point);
    }
}

// 优化地图，维护地图的规模
void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    for(auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();)
    {
        // 判断地图上的点在不在当前帧中，不在就删掉
        if(!curr_->isInFrame(iter->second->pos_))
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        // 定义匹配率，用匹配次数/可见次数，匹配率过低说明常见但是没有几次匹配
        // 或许是比较难识别的点，需要删掉
        float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
        if(match_ratio < map_point_erase_ratio_)
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        // 获取当前帧和地图点之间的夹角，角度过大则删除
        double angle = getViewAngle(curr_, iter->second);
        if(angle > M_PI / 6.)
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if(iter->second->good_ == false)
        {

        }
        iter++;
    }
    // 增加点的情况，如果当前帧在地图中匹配的点少于100时则增加点
    if(match_2dkp_index_.size() < 100)
        addMapPoints();
    // 如果点过多了，多于1000个，适当增加释放率
    if(map_->map_points_.size() > 1000)
    {
        map_point_erase_ratio_ += 0.05;
    }
    else{
        map_point_erase_ratio_ = 0.1;
    }
    cout<<"map_points: "<<map_->map_points_.size()<<endl;
}

double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
{
    // 相机中心指向空间点的坐标
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos(n.transpose() * point->norm_);
}

}