#ifndef FRAME_H
#define FRAME_H

#include "myslam/camera.h"
#include "myslam/ORBextractor.h"

#include <DBoW3/DBoW3.h>

namespace myslam
{

class KeyFrame;
class MapPoint;

class Frame
{

public:
    unsigned long           id_;
    string                  timeStamp_;
    Camera*                 camera_;
    SE3                     Tcw_;
    bool                    poseExist_;
    KeyFrame*               keyframe_trackRef_;
    
    vector<cv::KeyPoint>    keypoints_;
    vector<cv::KeyPoint>    unKeypoints_;
    
    vector<float>           depth_;
    vector<float>           uRight_;
    Mat                     descriptors_;
    vector<MapPoint*>       mappoints_;
    
    vector<float>           scaleFactors_;
    size_t                  N_;
    
    float                   xMin_;
    float                   xMax_;
    float                   yMin_;
    float                   yMax_;
    float                   gridPerPixelWidth_;
    float                   gridPerPixelHeight_;
    vector<int>             gridKeypoints_[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    vector<bool>            outliers_; 
    
    DBoW3::Vocabulary*      voc_;
    DBoW3::BowVector        bowVec_;
    DBoW3::FeatureVector    featVec_;
    
    Vector3d                Ow_;
    
    ORB_SLAM2::ORBextractor*  orb_;
    
    Frame(Mat &grayImg, Mat &depthImg, string timeStamp, Camera* camera, ORB_SLAM2::ORBextractor*  orb);
    
    void findDepth(Mat &depthImg);
    vector<Mat> getDespVector();
    bool isInFrame(MapPoint* mp);
    bool isInImg(const Vector2d &pixel);
    vector<int> getFeaturesInArea(const float &u, const float &v,const float &radius,
                                  const int min_level, const int max_level);
    void setPose(SE3 Tcw);
    void computeBow();
    
protected:
    void assignFeaturesToGrid();
    bool postionInGrad(const int &gradNumX, const int &gradNumY);
    void undistortKeyPoints();

};

}

#endif
