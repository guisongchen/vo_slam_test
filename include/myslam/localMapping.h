#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "myslam/visualOdometry.h"
#include <mutex>

namespace myslam
{
class LoopClosing;
    
class LocalMapping
{
public:
    Map*                map_;
    VisualOdometry*     vo_;
    LoopClosing*        loopCloser_;
    KeyFrame*           keyframe_curr_;
    KeyFrame*           keyframe_ref_;
    list<MapPoint*>     addedMapPoints_;
    
    list<KeyFrame*>     newKeyframes_;
    mutex               mutexNewKFs_;
    
    bool                acceptKeyFrameFlag_;
    mutex               mutexAccept_;
    
    bool                stopBAFlag_;
    bool                threadFinishFlag_;
    bool                finishRequestFlag_;
    mutex               mutexFinish_;
    
    bool                stopFlag_;
    bool                stopRequestFlag_;
    mutex               mutexStop_;
    
    LocalMapping(Map* map);
    
    void run();
    
    void insertKeyFrame(KeyFrame* keyframe);
    bool checkNewKeyFrames();
    int  inListKeyFrames();
    
    void setAcceptKeyFrame(bool flag);
    bool getAcceptStatus();
    void interruptBA();
    
    void setFinish();
    bool checkFinish();
    void requestFinish();
    bool checkFinishRequest();
    
    void requestStop();
    bool checkStopRequst();
    bool checkStopState();
    bool isStopped();
    void release();
    
protected:
    void processNewKeyFrame();
    void createNewMapPoints();
    void searchInNeighbors();
    void cullingKeyFrames();
    void cullingMapPoints();
    
    Eigen::Matrix3d computeF12(SE3 &T1w, SE3 &T2w, Eigen::Matrix3d &K);
};
    
}

#endif
