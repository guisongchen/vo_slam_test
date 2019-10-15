#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/config.h"

namespace myslam
{

Camera::Camera()
{
    fx_   = Config::get<float>("camera_fx");
    fy_   = Config::get<float>("camera_fy");
    cx_   = Config::get<float>("camera_cx");
    cy_   = Config::get<float>("camera_cy");
    bf_   = Config::get<float>("camera_bf");
    fps_  = Config::get<int>("camera_fps");
    depth_scale_ = Config::get<float> ("camera_depthScale");
    thDepth_ = Config::get<float> ("thDepth");
    
    b_  = bf_ / fx_;
    thDepth_ *= b_ ;
    K_ = (cv::Mat_<float>(3,3) << fx_, 0,   cx_,
                                  0,   fy_, cy_,
                                  0,   0,   1);
    
    K_eigen_ << static_cast<double>(fx_), 0.0, static_cast<double>(cx_),
                0.0, static_cast<double>(fy_), static_cast<double>(cy_),
                0.0, 0.0, 1.0;
                
    Mat distCoef(4, 1, CV_32F);
    distCoef.at<float>(0) = Config::get<float>("camera_k1");
    distCoef.at<float>(1) = Config::get<float>("camera_k2");
    distCoef.at<float>(2) = Config::get<float>("camera_p1");
    distCoef.at<float>(3) = Config::get<float>("camera_p2");
    const float k3 = Config::get<float>("camera_k3");
    if (k3!=0)
    {
        distCoef.resize(5);
        distCoef.at<float>(4) = k3;
    }
    distCoef.copyTo(distCoef_);
    
    xMax_ = Config::get<float>("camera_width");
    yMax_ = Config::get<float>("camera_height");
    xMin_ = 0.0f;
    yMin_ = 0.0f;
    
    gridPerPixelWidth_ = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(xMax_-xMin_);
    gridPerPixelHeight_ = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(yMax_-yMin_);
    
}

void Camera::printCameraInfo()
{
    cout << "fx: " << fx_ << "\n"
         << "fy: " << fy_ << "\n"
         << "cx: " << cx_ << "\n"
         << "cy: " << cy_ << "\n"
         << "bf: " << bf_ << "\n" 
         << "fps: " << fps_ << "\n" 
         << "depthScale: " << depth_scale_ << "\n"
         << "thDepth: " << thDepth_ << endl;
}

Vector3d Camera::pixel2camera(const Vector2d& p, double depth)
{
    return Vector3d (
        (p[0] - cx_) * depth / fx_,
        (p[1] - cy_) * depth / fy_,
        depth);
}

Vector2d Camera::camera2pixel(const Eigen::Vector3d& p3d)
{
    return Vector2d ( fx_ * p3d[0] / p3d[2] + cx_, fy_ * p3d[1] / p3d[2] + cy_);
}

Vector3d Camera::pixel2world(const Vector2d& p, double depth, SE3& T_c_w)
{
    return T_c_w.inverse() * pixel2camera(p, depth);
}

Vector3d Camera::pixel2world(const cv::KeyPoint &kpt, const float z, SE3 &T_c_w)
{
    return T_c_w.inverse() * pixel2camera(kpt, z);
}

Vector3d Camera::pixel2camera(const cv::KeyPoint &kpt, const float z)
{
    const float u = kpt.pt.x;
    const float v = kpt.pt.y;
    const float x = (u - cx_)*z / fx_;
    const float y = (v - cy_)*z / fy_;
    
    return Vector3d(x, y, z);
}

Vector2d Camera::world2pixel(const Eigen::Vector3d& p, Sophus::SE3& T_c_w)
{
    return camera2pixel(T_c_w * p);
}
    
}
