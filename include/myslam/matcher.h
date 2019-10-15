#ifndef MATCHER_H
#define MATCHER_H

#include "myslam/keyframe.h"

namespace myslam
{

class Matcher
{
public:
    Matcher() {}
    ~Matcher() {}
    Matcher(float ratio);
    
    int searchByProjection(Frame *frame_curr, Frame *frame_last, const float radius, bool checkRot = true);
    int searchByProjection(Frame *frame_curr, KeyFrame *keyframe, const float radius, const float distThreshold,
                           const set<MapPoint*> &found, bool checkRot = true);
    int searchByProjection(Frame *frame, const vector<MapPoint*> &mappoints, const float thRadius);
    int searchByProjection(KeyFrame* keyframe, Sophus::Sim3 &Scw, vector<MapPoint*> &loopMapPoints,
                           vector<MapPoint*> &matchMapPoints, int th);
    
    int searchByBoW(KeyFrame* keyframe, Frame* frame, vector<MapPoint*> &mappointMatches, bool checkRot = true);
    int searchByBoW(KeyFrame* keyframe1, KeyFrame* keyframe2, vector<MapPoint*> &mappointMatches, bool checkRot);
    
    int searchBySim3(KeyFrame* keyframe1, KeyFrame* keyframe2, vector<MapPoint*> &matches12, 
                     Sophus::Sim3 &S12, const float th);
    
    static int computeDistance(const Mat &desp1, const Mat &desp2);
    
    int searchForTriangulation(KeyFrame *keyframe1, KeyFrame *keyframe2, vector< pair<int, int> > &matchIdxs);
    int searchForTriangulation(KeyFrame *keyframe1, KeyFrame *keyframe2, vector< pair<int, int> > &matchIdxs,
                               Eigen::Matrix3d &F12, bool checkRot = true);
    
    int fuseMapPoints(KeyFrame* keyframe, vector<MapPoint*> &mappoints, const float &threshold);
    int fuseByPose(KeyFrame* keyframe, Sophus::Sim3 &Scw, vector<MapPoint*> &loopMapPoints,
                   vector<MapPoint*> &replaceMapPoints, const float th);

protected:
    void computeThreeMax(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    bool checkEpipolarConstrain(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
                                const Eigen::Matrix3d &F, const KeyFrame *keyframe2);
private:
    float ratio_;
};
    
}

#endif
