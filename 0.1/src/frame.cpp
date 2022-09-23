#include "myslam/frame.h"

namespace myslam
{
Frame::Frame(): id_(-1), time_stamp_(-1), camera_(nullptr)
{

}

Frame::Frame(long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth):
    id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
{

}

Frame::~Frame()
{

}

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;
    return Frame::Ptr(new Frame(factory_id++));
}

double Frame::findDepth(const cv::KeyPoint& kp)
{
    int x = cvRound(kp.pt.x); // 关键点坐标
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];    // RGB_D相机图像深度
    if(d != 0)
    {
        return double(d) / camera_->depth_scale_;
    }
    else
    {
        // 检查周围的像素
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        for(int i = 0; i < 4; ++i)
        {
            d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
            if(d != 0)
            {
                return double(d) / camera_->depth_scale_;
            }
        }
    }
    return -1.0;
}

Vector3d Frame::getCamCenter() const
{
    // translation()表示求SE3的平移向量，所以这里实际求的是相机坐标系下的(0,0,0)在世界坐标系下的坐标
    return T_c_w_.inverse().translation();  // = (R^T)*-t
}

bool Frame::isInFrame(const Vector3d& pt_world)
{
    Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
    if(p_cam(2, 0) < 0)  // z是相机与物体的距离，如果z小于0显然返回false
        return false;
    Vector2d pixel = camera_->world2pixel(pt_world, T_c_w_);
    return pixel(0, 0) > 0 && pixel(1, 0) > 0
        && pixel(0, 0) < color_.cols
        && pixel(1, 0) < color_.rows;
}

} // namespace myslam