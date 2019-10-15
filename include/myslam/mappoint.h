#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"
#include <mutex>

namespace myslam
{
class Frame;
class KeyFrame;
class Map;

class MapPoint
{
public:
    unsigned long   id_;
    Vector3d        pos_;
    Vector3d        normalVector_;
    
    KeyFrame*       keyFrame_ref_;
    MapPoint*       replacedMapPoint_;
    Map*            map_;
    size_t          idx_;
    bool            trackInLocalMap_;
    Mat             descriptor_;
    
    int             found_cnt_;  
    int             visible_cnt_; 
    int             observe_cnt_; 
    
    unsigned long   trackIdxOfFrame_;
    unsigned long   visualIdxOfFrame_;
    unsigned long   loopPointIdxOfKF_;
    unsigned long   loopCorrectByKF_;
    unsigned long   correctReference_;
    unsigned long   localBAKFId_;
    
    float           minDistance_;
    float           maxDistance_;
    int             trackScaleLevel_;
    long int        firstAddedIdxofKF_;
    
    float           trackProj_u_;
    float           trackProj_uR_;
    float           trackProj_v_;
    float           viewCos_;
    unsigned long   fuseForKF_;
    
    map<KeyFrame*, size_t> observedKFs_;

    mutex           mutexFeature_;
    mutex           mutexPose_;
    static mutex    mutexOptimizer_;
    
    bool            badFlag_;
    
    
    MapPoint(const Vector3d& position, Frame* frame, size_t idx, Map* map);
    MapPoint(const Vector3d& position, KeyFrame* keyframe, size_t idx, Map* map);
    
    map<KeyFrame*, size_t> getObservedKFs();
    
    void addFound(const int cnt=1);
    void addVisible(const int cnt=1);
    void addObservation(KeyFrame* keyframe, size_t idx);
    
    bool beObserved(KeyFrame* kf);
    
    void updateNormalAndDepth();
    void computeDescriptor();
    
    int predictScale(const float &currDist, Frame* frame);
    int predictScale(const float &currDist, KeyFrame* kf);
    
    void replaceMapPoint(MapPoint* mp);
    MapPoint* getReplacedMapPoint();
    
    int getIndexInKeyFrame(KeyFrame* keyframe);
    Vector3d getPose();
    void setPose(const Vector3d &pos);
    Mat getDescriptor();
    Vector3d getNormalVector();
    int getObsCnt();
    float getFoundRatio();
    
    void eraseObservedKF(KeyFrame* kf);
    void eraseMapPoint();
    
    bool isBad(); // bad mappoint, need to be erased
    
    float getMinDistanceThreshold();
    float getMaxDistanceThreshold();

};

    
}

#endif
