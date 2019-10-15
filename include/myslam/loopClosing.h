#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "myslam/map.h"
#include "myslam/keyframe.h"
#include "myslam/localMapping.h"

#include <DBoW3/DBoW3.h>
#include <sophus/sim3.h>
#include <mutex>
#include <condition_variable>

namespace myslam
{

class LoopClosing 
{
public:
    typedef pair<set<KeyFrame*>, int> consistentGroup;
    typedef map<KeyFrame*, Sophus::Sim3, less<KeyFrame*>,
        Eigen::aligned_allocator< pair<const KeyFrame*, Sophus::Sim3> > > KeyFrameAndPose;

    unsigned long       lastLoopKFId_;
    
    Map*                map_;
    DBoW3::Vocabulary*  voc_;
    KeyFrame*           keyframe_curr_;
    KeyFrame*           keyframe_match_;
    
    vector<consistentGroup> prevConsistentGroups_; 
    vector<KeyFrame*>   enoughConCandidates_;
    
    list<KeyFrame*>     newKeyFrames_;
    mutex               mutexNewKFs_;
    condition_variable  condNewKFs_;
    
    vector<MapPoint*>   matchMapPoints_;
    vector<MapPoint*>   loopConnectKFMapPoints_;
    vector<KeyFrame*>   currConnectKFs_;
    
    Sophus::Sim3        Scw_;
    
    bool                finishFlag_;
    bool                requestFinishFlag_;
    mutex               mutexFinish_;
    
    bool                fixScaleFlag_;
    LocalMapping*       localMapper_;
    
    LoopClosing(Map* map);
    
    void run();
    
    bool checkNewKeyFrames();
    void insertKeyFrame(KeyFrame* keyframe);
    
    bool checkFinishRequest();
    void requestFinish();
    bool checkFinish();
    void setFinish();
    
    void setLocalMapper(LocalMapping* localMapper);
    void setVocabulary(DBoW3::Vocabulary* voc);

protected:
    bool detectLoop();
    bool computeSim3();
    void searchAndFuse(const KeyFrameAndPose &correctPose);
    void correctLoop();
    
};
    
}

#endif
