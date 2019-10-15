#ifndef CAMERA_H
#define CAMERA_H

#include "myslam/common_include.h"

namespace myslam
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    
class Camera
{
public:
    float       fx_;
    float       fy_;
    float       cx_;
    float       cy_;
    float       depth_scale_;
    float       bf_;
    float       thDepth_;
    float       b_;
    
    int         fps_;
    Mat         K_;
    Matrix3d    K_eigen_;
    Mat         distCoef_;
    
    float       xMin_;
    float       xMax_;
    float       yMin_;
    float       yMax_;

    float       gridPerPixelWidth_;
    float       gridPerPixelHeight_;
    
    
    Camera();
    
    void printCameraInfo();
    
    Vector3d pixel2camera (const Vector2d& p, double depth=1);
    Vector2d camera2pixel(const Vector3d& p3d);
    Vector3d pixel2world(const Vector2d& p, double depth, SE3& T_c_w);
    Vector2d world2pixel(const Vector3d& p, SE3& T_c_w);
    
    Vector3d pixel2world(const cv::KeyPoint &kpt, const float z, SE3 &T_c_w);
    Vector3d pixel2camera(const cv::KeyPoint &kpt, const float z);
};
    
    
}

#endif
