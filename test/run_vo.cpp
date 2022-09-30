// --------------- 测试视觉里程计-----------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout<<"Usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile(argv[1]);
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin(dataset_dir + "/associate.txt");
    if(!fin)
    {
        cout<<"Please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    // 定义图片名数组和时间戳数组
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        // atof()把字符串转换成浮点数
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if(!fin.good())
            break;
    }
    // 创建相机
    myslam::Camera::Ptr camera(new myslam::Camera);

    // 可视化，viz模块
    // 1.创建窗口
    cv::viz::Viz3d vis("Visual Odometry");
    // 2.创建坐标系部件，参数是坐标长度
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    // 渲染属性，第一个参数是枚举，(线宽, 数值)
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    // showWidget函数将部件添加到窗口内
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);
    // 3.(可选)设置视角，相机位置坐标，相机焦点坐标，相机y轴朝向
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    // 视角位姿
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    // 设置观看视角
    vis.setViewerPose(cam_pose);
    cout<<"read total "<<rgb_files.size()<<" entries"<<endl;
    // 画面的快速刷新呈现动态，由循环控制
    for(int i = 0; i < rgb_files.size(); ++i)
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i], -1);
        if(color.data == nullptr || depth.data == nullptr)
            break;
        // 创建帧
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];
        // 每帧的运算时间，看实时性
        boost::timer timer;
        // 将帧添加进去
        vo->addFrame(pFrame);
        cout<<"VO costs time: "<<timer.elapsed()<<endl;

        if(vo->state_ == myslam::VisualOdometry::LOST)
            break;
        // 可视化窗口动的是相机坐标系，求相机坐标系下的点在世界坐标系下的坐标
        SE3 Tcw = pFrame->T_c_w_.inverse();
        
        // T
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.rotation_matrix()(0, 0), Tcw.rotation_matrix()(0, 1), Tcw.rotation_matrix()(0, 2),
                Tcw.rotation_matrix()(1, 0), Tcw.rotation_matrix()(1, 1), Tcw.rotation_matrix()(1, 2),
                Tcw.rotation_matrix()(2, 0), Tcw.rotation_matrix()(2, 1), Tcw.rotation_matrix()(2, 2)
            ),
            cv::Affine3d::Vec3(
                Tcw.translation()(0, 0), Tcw.translation()(1, 0), Tcw.translation()(2, 0)
            )
        );

        // 地图点的投影
        Mat img_show = color.clone();
        for(auto& pt : vo->map_->map_points_)
        {
            myslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel(p->pos_, pFrame->T_c_w_);
            cv::circle(img_show, cv::Point2f(pixel(0, 0), pixel(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("image", color);
        cv::waitKey(1);
        // viz可视化窗口
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
    }
    return 0;
}   