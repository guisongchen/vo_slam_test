#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/keyframe.h"
#include <unordered_map>
#include <DBoW3/DBoW3.h>
#include <mutex>

namespace myslam
{
    
class Map
{
public:
    set<MapPoint*>                mappoints_;
    set<KeyFrame*>                keyframes_;
    
    vector<MapPoint*>             localMapPoints_;
    
    vector< list<KeyFrame*> >     invertIdxs_;
    vector<Frame*>                lostFrames_;
    DBoW3::Vocabulary*            voc_;
    
    mutex                         mutexMap_;
    mutex                         mutexMapUpdate_;
    
    unsigned long                 maxKFId_;
    
    bool                          saveVocabularyFlag_;
    
    mutex                         mutexCreateMapPoint_;
     
    Map();
     
    void insertKeyFrame(KeyFrame* keyframe);
    void eraseKeyFrame(KeyFrame* keyframe);
    
    void insertMapPoint(MapPoint* mappoint);
    void eraseMapPoint(MapPoint* mappoint);
    
    void createVocabulary();
    double score(const DBoW3::BowVector &v1, const DBoW3::BowVector &v2) const;
    
    void setLocalMapPoints(const vector<MapPoint*> mappoints);
    vector<MapPoint*> getLocalMapPoints();
    
    vector<KeyFrame*> getAllKeyFrames();
    unsigned long getAllKeyFramesCnt();
    
    vector<MapPoint*> getAllMapPoints();
    unsigned long getAllMapPointsCnt();
    
    vector<KeyFrame*> detectRelocalizationCandidates(Frame* frame);
    vector<KeyFrame*> detectLoopCandidates(KeyFrame* keyframe, float minScore);
     
};
    
}

#endif
