#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam
{

class KeyFrame
{
public:
    unsigned long           id_;
    unsigned long           orgFrameId_;
    unsigned long           trackFrameId_;
    string                  timeStamp_;

    Camera*                 camera_;
    SE3                     Tcw_;
    SE3                     Tcp_;
    Vector3d                Ow_;
    bool                    firstConnect_;
    KeyFrame*               parent_;
    set<KeyFrame*>          children_;
    
    vector<cv::KeyPoint>    keypoints_;
    vector<cv::KeyPoint>    unKeypoints_;
    
    vector<float>           depth_;
    vector<float>           uRight_;
    Mat                     descriptors_;
    vector<MapPoint*>       mappoints_;
    
    map<KeyFrame*, int>     connectedKFWts_;
    vector<KeyFrame*>       orderedConnectKFs_;
    vector<int>             orderedWTs_;
    
    vector<float>           scaleFactors_;
    size_t                  N_;
    unsigned long           fuseKFId_;
    
    float                   xMin_;
    float                   xMax_;
    float                   yMin_;
    float                   yMax_;
    float                   gridPerPixelWidth_;
    float                   gridPerPixelHeight_;
    vector< vector< vector<int> > >  gridKeypoints_;
    
    bool                    badFlag_;
    
    DBoW3::Vocabulary*      voc_;
    DBoW3::BowVector        bowVec_;
    DBoW3::FeatureVector    featVec_;
    
    unsigned long           relocateFrameId_;
    int                     relocateWordCnt_;
    float                   relocateScore_;

    unsigned long           loopKFId_;
    int                     loopWordCnt_;
    float                   loopScore_;
    set<KeyFrame*>          loopEdges_;
    bool                    notEraseLoopDetecting_;
    bool                    toBeEraseAfterLoopDetect_;
    
    mutex                   mutexConnection_;
    mutex                   mutexFeature_;
    mutex                   mutexPose_;
    
    unsigned long           localBAKFId_;
    unsigned long           BAFixId_;
    
    Map*                    map_;
    
    KeyFrame(Frame* frame, Map* map);

    SE3 getPose();
    void setPose(SE3 &Tcw);
    
    bool isInImg(const float &u, const float &v);
    Vector3d getCamCenter();
    void updateConnections();
    
    vector<KeyFrame*> getBestCovisibleKFs(const int &N);
    vector<KeyFrame*> getCovisiblesByWeight(const int &w);
    
    int trackedMapPoints(const int &minObs);
    double computeMidDepth();
    
    vector<cv::KeyPoint> getKeyPoints();
    vector<MapPoint*> getMapPoints();
    set<KeyFrame*> getChildren();
    KeyFrame* getParent();
    vector<int> getFeaturesInArea(const float &u, const float &v, const float &radius);
    set<KeyFrame*> getConnectKFs();
    vector<KeyFrame*> getOrderedKFs();
    int getWeight(KeyFrame* keyframe);
    
    void addMapPoint(MapPoint* mp, const size_t &idx);
    void replaceMapPoint(MapPoint* mp, const size_t &idx);
    void setMapPointNull(const size_t &idx);
    
    vector<Mat> getDespVector();
    void computeBow();
    void eraseKeyFrame();
    void eraseConnection(KeyFrame* kf);
    
    void setParent(KeyFrame* kf);
    void setChild(KeyFrame* kf);
    void eraseChild(KeyFrame* kf);
    void addLoopEdge(KeyFrame* kf);
    bool isBad();
    
    void setNotEraseLoopDetectingKF();
    void setEraseLoopDetectingKF();
    
protected:
    void addConnection(KeyFrame *kf, const int &wt);
    void updateBestCovisibles();
    
};

}

#endif
