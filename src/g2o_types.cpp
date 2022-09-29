#include "myslam/g2o_types.h"

namespace myslam
{

// 1.计算重投影误差
void EdgeProjectXYZ2UVPoseOnly::computeError()
{
    // 相机位姿为顶点
    const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    // 误差是测量值减去估计值，估计是为T*p
    _error = _measurement - camera_->camera2pixel(pose->estimate().map(point_));
}

// 2.计算线性增量函数，雅可比矩阵
void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
    // 顶点取出位姿
    g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
    // 位姿构造四元数形式T
    g2o::SE3Quat T(pose->estimate());
    // 变换后的3D点xyz坐标
    Vector3d xyz_trans = T.map(point_);
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z * z;
    // 雅可比矩阵2*6
    _jacobianOplusXi(0, 0) = x*y/z_2 * camera_->fx_;
    _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx_;
    _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
    _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
    _jacobianOplusXi ( 0,4 ) = 0;
    _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;

    _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
    _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
    _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
    _jacobianOplusXi ( 1,3 ) = 0;
    _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
    _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;
}

}