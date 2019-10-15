#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/keyframe.h"
#include "myslam/map.h"
#include "myslam/ORBextractor.h"

#include <DBoW3/DBoW3.h>

namespace myslam
{
class LocalMapping;
class Drawer;
    
class VisualOdometry
{
public:
    enum VOState {
        INITILIZING,
        OK,
        LOST
    };
    
    VOState                 state_;
    Map*                    map_;
    LocalMapping*           localMapper_;
    KeyFrame*               keyframe_trackRef_;
    Frame*                  frame_curr_;
    Frame*                  frame_last_;
    SE3                     Tcl_;
    bool                    motionModel_;
    
    unsigned long           lastKeyFrameId_;
    unsigned long           lastRelocateFrameId_;
    
    int                     maxFrameGap_;
    int                     inliers_num_;
    int                     num_lost_;
    int                     max_lost_;
    
    int                     num_of_features_;
    float                   scale_factor_;
    int                     level_pyramid_;
    int                     patch_size_;
    
    vector<KeyFrame*>       localKeyframes_;
    vector<MapPoint*>       localMappoints_;
    list<MapPoint*>         tempMappoints_;
    
    DBoW3::Vocabulary*      voc_;
    
    Camera*                 camera_;
    bool                    rgbFormat_;
    
    Mat                     grayImg_;
    
    Drawer*                 drawer_;
    VOState                 lastState_;
    
    string                  timeStamp_;
    
    list<KeyFrame*>         trackRefKeyFrameDB_;
    list<SE3>               TcrDB_;
    list<bool>              stateDB_;
    list<string>            timestampDB_;

    ORB_SLAM2::ORBextractor*  orb_;

    VisualOdometry(Map* map, Camera* camera, Drawer* drawer);
    ~VisualOdometry();
    
    void run(Mat &rgbImg, Mat &depthImg, string &timestamp);
    SE3 poseRtToSE3(const Mat& R, const Mat& t);
    void printPose(SE3 &Tcw);
    
protected:
    void createFrame(Mat &rgbImg, Mat &depthImg, string timeStamp);

    bool trackWithMotion();
    bool trackRefKeyFrame();
    
    void recoverLastFrame();
    void updateLastFrame();
    bool trackLocalMap();
    void updateLocalMap();
    void updateLocalKeyFrames();
    void updateLocalMapPoints();
    void searchLocalMapPoints();
    void cullingTempMapPoints();
    
    int  poseEstimateByPnP(Frame* frame, set<MapPoint*> &found, vector<MapPoint*> &mappointMatches);
    bool relocalization();
    bool checkEstimatePose();
    bool needNewKeyFrame();
    void createNewKeyFrame();
    void initVisualOdometry();
    
    int cullingOutliersBeforeLocalMap();
    void cullingOutliersOfFrame();

};
    
}

#endif
