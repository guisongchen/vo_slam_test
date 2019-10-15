#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include "myslam/keyframe.h"

namespace myslam
{

class Sim3Solver
{
public:
    Sim3Solver(KeyFrame* keyframe1, KeyFrame* keyframe2, vector<MapPoint*> &matches12, const bool fixScale=true);
    void setRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);
    Sophus::Sim3 iterate(int iterations_req, bool &stopFlag, bool &emptyFlag,
                         vector<bool> &inlierFlags, int &inliers_cnt);

    Matrix3d getEstimatedRotation();
    Vector3d getEstimatedTranslation();
    double getEstimatedScale();

protected:
    void computeSim3(Matrix3d &P1, Matrix3d &P2);
    void checkInliers();
    void project(vector<Vector3d> &p3ds, vector<Vector2d> &p2ds, const Sophus::Sim3 &Scw);    
    int randomInt(int min, int max);

protected:
    KeyFrame*               keyframe1_;
    KeyFrame*               keyframe2_;
    vector<MapPoint*>       matches12_;
    bool                    fixScale_;

    int                     matches_cnt_;
    vector<MapPoint*>       mappoints1_;
    vector<MapPoint*>       mappoints2_;
    vector<Vector3d>        pcams1_;
    vector<Vector3d>        pcams2_;
    vector<Vector2d>        pixels1_;
    vector<Vector2d>        pixels2_;
    vector<int>             maxError1_;
    vector<int>             maxError2_;
    vector<int>             matchedIndexs_;
    
    vector<bool>            inlierFlags_;
    vector<bool>            inlierFlags_best_;
    int                     inliers_cnt_;
    int                     inliers_best_;
    double                  s12_best_;
    Sophus::Sim3            T12_best_;
    Matrix3d                R12_best_;
    Vector3d                t12_best_;
    
    int                     iterations_global_;              
    double                  ransacProb_;
    int                     ransacInlierThreshold_;
    int                     ransacMaxIters_;
    vector<int>             idxForRandom_;
    
    
    double                  s12_;
    Sophus::Sim3            T12_;
    Matrix3d                R12_;
    Vector3d                t12_;
    Sophus::Sim3            T21_;

};

} 

#endif 

