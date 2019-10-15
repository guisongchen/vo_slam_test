#ifndef DRAWER_H
#define DRAWER_H

#include "myslam/map.h"
#include "myslam/visualOdometry.h"

#include <pangolin/pangolin.h>
#include <mutex>

namespace myslam
{
    
class Drawer
{
public:
    Drawer(Map* map_curr);
    
    Map*        map_curr_;
    
    int         width_;
    int         height_;
    double      fu_;  // camera matrix intrinsic
    double      fv_;  // camera matrix intrinsic
    double      u0_;  // camera matrix intrinsic
    double      v0_;  // camera matrix intrinsic
    double      viewpointX_;
    double      viewpointY_;
    double      viewpointZ_;
    
    SE3         Tcw_;
    
    mutex       mutexPose_;
    mutex       mutexFrame_;
    
    Mat         img_curr_;
    vector<cv::KeyPoint> keypoints_curr_;
    vector<bool>         inMapFlag_;
    vector<bool>         inVOFlag_;
    
    size_t               N_;
    int                  mapTrackedCnt_;
    int                  voTrackedCnt_;
    int                  state_;
    
    bool                 finishFlag_;
    bool                 finishRequestFlag_;
    mutex                mutexFinish_;
    
    void run();

    void setCurrPose(SE3 &Tcw);
    void updateCurrFrame(VisualOdometry* vo_curr);
    
    void setFinish();
    void requestFinish();
    bool checkFinish();
    bool checkFinishRequest();

protected:
    void getOpenGLCameraMatrix (pangolin::OpenGlMatrix &M);
    void drawMapPoints();
    void drawKeyFrames();
    void drawCamera(pangolin::OpenGlMatrix &Twc);
    void drawTextOnImg(Mat &frameImg, Mat &textImg, int state);
    Mat drawCurrFrameImg();

};

}

#endif
