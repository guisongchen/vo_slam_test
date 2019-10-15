#include "myslam/sim3Solver.h"
#include <Eigen/Eigenvalues>
#include <cmath>

namespace myslam
{

Sim3Solver::Sim3Solver(KeyFrame* keyframe1, KeyFrame* keyframe2, 
                       vector<MapPoint*> &matches12, const bool fixScale)
    : keyframe1_(keyframe1), keyframe2_(keyframe2), matches12_(matches12), fixScale_(fixScale),
      inliers_best_(0), iterations_global_(0) 
{
    vector<MapPoint*> queryMappoints1 = keyframe1->getMapPoints();
    
    matches_cnt_ = matches12.size();
    mappoints1_.reserve(matches_cnt_);
    mappoints2_.reserve(matches_cnt_);
    pcams1_.reserve(matches_cnt_);
    pcams2_.reserve(matches_cnt_);
    matchedIndexs_.reserve(matches_cnt_);
    idxForRandom_.reserve(matches_cnt_);
    
    const SE3 Tcw1 = keyframe1->Tcw_;
    const SE3 Tcw2 = keyframe2->Tcw_;
    
    Camera* camera = keyframe1->camera_;
    
    int cnt = 0;
    for (int i = 0; i < matches_cnt_; i++)
    {
        if(matches12[i])
        {
            MapPoint* mp1 = queryMappoints1[i];
            MapPoint* mp2 = matches12[i];

            if(!mp1)
                continue;

            if(mp1->isBad() || mp2->isBad())
                continue;
            
            int idx1 = mp1->getIndexInKeyFrame(keyframe1);
            int idx2 = mp2->getIndexInKeyFrame(keyframe2);

            if(idx1 < 0 || idx2 < 0)
                continue;

            const cv::KeyPoint &kp1 = keyframe1->unKeypoints_[idx1];
            const cv::KeyPoint &kp2 = keyframe2->unKeypoints_[idx2];

            const float sigma1 = keyframe1->scaleFactors_[kp1.octave];
            const float sigma2 = keyframe2->scaleFactors_[kp2.octave];
            maxError1_.push_back(9.210*sigma1*sigma1);
            maxError2_.push_back(9.210*sigma2*sigma2);
            
            mappoints1_.push_back(mp1);
            mappoints2_.push_back(mp2);
            matchedIndexs_.push_back(i);
            
            Vector3d pcam1 = Tcw1 * mp1->getPose();
            Vector3d pcam2 = Tcw2 * mp2->getPose();
            
            pcams1_.push_back(pcam1);
            pcams2_.push_back(pcam2);
            
            pixels1_.push_back(camera->camera2pixel(pcam1));
            pixels2_.push_back(camera->camera2pixel(pcam2));
            
            idxForRandom_.push_back(cnt++);
        }
    }

    setRansacParameters();
}

void Sim3Solver::setRansacParameters(double probability, int minInliers, int maxIterations)
{
    
    ransacProb_ = probability;
    ransacInlierThreshold_ = minInliers;
    ransacMaxIters_ = maxIterations;
    
    const int N  = mappoints1_.size();
    inlierFlags_.resize(N);

    const float epsilon = static_cast<float>(ransacInlierThreshold_) / static_cast<float>(N);
    int iterations;
    if(ransacInlierThreshold_ == N)
        iterations = 1;
    else
        iterations = ceil(log(1 - ransacProb_)/log(1 - pow(epsilon,3)));

    ransacMaxIters_ = max(1, min(iterations, maxIterations));
    
    iterations_global_ = 0;
}

Sophus::Sim3 Sim3Solver::iterate(int iterations_req, bool &stopFlag, bool &emptyFlag,
                                 vector<bool> &inlierFlags, int &inliers_cnt)
{
    stopFlag = false;
    emptyFlag = false;
    inlierFlags = vector<bool>(matches_cnt_,false);
    inliers_cnt=0;

    if (mappoints1_.size() < ransacInlierThreshold_)
    {
        stopFlag = true;
        emptyFlag = true;
        
        return Sophus::Sim3();
    }

    vector<int> availableIdxs;
    
    Eigen::Matrix3d points_coordinate1;
    Eigen::Matrix3d points_coordinate2;

    int iterations_curr = 0;
    while(iterations_global_ < ransacMaxIters_ && iterations_curr < iterations_req)
    {
        iterations_curr++;
        iterations_global_++;

        availableIdxs = idxForRandom_;

        // Get min set of points
        for(int i = 0; i < 3; ++i)
        {
            int randi = randomInt(0, availableIdxs.size()-1);
            int idx = availableIdxs[randi];

            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            points_coordinate1.block<3,1>(0,i) = pcams1_[idx];
            points_coordinate2.block<3,1>(0,i) = pcams2_[idx];
            
            availableIdxs[randi] = availableIdxs.back();
            availableIdxs.pop_back();
        }

        computeSim3(points_coordinate1, points_coordinate2);
        checkInliers();

        if (inliers_cnt_ >= inliers_best_)
        {
            inliers_best_ = inliers_cnt_;
            inlierFlags_best_ = inlierFlags_;
            
            T12_best_ = T12_;
            R12_best_ = R12_;
            t12_best_ = t12_;
            s12_best_ = s12_;

            if(inliers_cnt_ > ransacInlierThreshold_)
            {
                inliers_cnt = inliers_cnt_;
                
                for(int i = 0, N = mappoints1_.size(); i < N; i++)
                    if(inlierFlags_[i])
                        inlierFlags[matchedIndexs_[i]] = true;
                
//                 cout << "returned successed Sim3:\n" << T12_best_ << endl;
                    
                return T12_best_;
            }
        }
    }

    if( iterations_global_ >= ransacMaxIters_)
        stopFlag=true;
    
    emptyFlag = true;

    return Sophus::Sim3();
}

void Sim3Solver::computeSim3(Eigen::Matrix3d &P1, Eigen::Matrix3d &P2)
{
    // Step 1: Centroid and relative coordinates
    Vector3d O1 = P1.rowwise().mean();
    Vector3d O2 = P2.rowwise().mean();
    Eigen::Matrix3d Pr1 = P1.colwise() - O1;
    Eigen::Matrix3d Pr2 = P2.colwise() - O2;
    
    // Step 2: Compute M matrix
    Eigen::Matrix3d M = Pr2 * Pr1.transpose();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;
    Eigen::Matrix4d N;

    N11 = M(0,0) + M(1,1) + M(2,2);
    N12 = M(1,2) - M(2,1);
    N13 = M(2,0) - M(0,2);
    N14 = M(0,1) - M(1,0);
    N22 = M(0,0) - M(1,1) - M(2,2);
    N23 = M(0,1) + M(1,0);
    N24 = M(2,0) + M(0,2);
    N33 = -M(0,0) + M(1,1) - M(2,2);
    N34 = M(1,2) + M(2,1);
    N44 = -M(0,0) - M(1,1) + M(2,2);

    N << N11, N12, N13, N14,
         N12, N22, N23, N24,
         N13, N23, N33, N34,
         N14, N24, N34, N44;
         
    // Step 4: Eigenvector of the highest eigenvalue
    Eigen::EigenSolver<Eigen::Matrix4d> solver(N);
    Eigen::Matrix4d evec = solver.eigenvectors().real(); // normlized
    Eigen::Vector4d eval = solver.eigenvalues().real();
    
    int maxColIdx;
    eval.maxCoeff(&maxColIdx);
    
    // vec is the unit quaternion of the desired rotation(w, x, y, z)
    Eigen::Vector4d vec = evec.col(maxColIdx);
    Eigen::Quaterniond q(vec[0], vec[1], vec[2], vec[3]); // init as wxyz, store&out as xyzw
    R12_ = q.toRotationMatrix();

    // Step 5: Rotate set 2
    Eigen::Matrix3d P3 = R12_ * Pr2;

    // Step 6: Scale
    if(!fixScale_)
    {
        double nom = (Pr1.array()*P3.array()).sum();
        double den = P3.squaredNorm();
        s12_ = nom / den;
    }
    else
        s12_ = 1.0;

    t12_ = O1 - s12_* R12_ * O2;
    
    T12_ = Sophus::Sim3(Sophus::ScSO3(s12_*R12_), t12_);
    T21_ = T12_.inverse();
}

void Sim3Solver::checkInliers()
{
    vector<Vector2d> p2ds1;
    vector<Vector2d> p2ds2;
    
    project(pcams2_, p2ds1, T12_);
    project(pcams1_, p2ds2, T21_);
    
    inliers_cnt_ = 0;

    for (int i = 0, N = pixels1_.size(); i < N; i++)
    {
        Vector2d dist1 = pixels1_[i] - p2ds1[i];
        Vector2d dist2 = pixels2_[i] - p2ds2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if (err1 < maxError1_[i] && err2 < maxError2_[i])
        {
            inlierFlags_[i] = true;
            inliers_cnt_++;
        }
        else
            inlierFlags_[i] = false;
    }
}


Eigen::Matrix3d Sim3Solver::getEstimatedRotation()
{
    return R12_best_;
}

Vector3d Sim3Solver::getEstimatedTranslation()
{
    return t12_best_;
}

double Sim3Solver::getEstimatedScale()
{
    return s12_best_;
}

void Sim3Solver::project(vector<Vector3d> &p3ds, vector<Vector2d> &p2ds, const Sophus::Sim3& Scw)
{    
    Camera* camera = keyframe1_->camera_;
    const float &fx = camera->fx_;
    const float &fy = camera->fy_;
    const float &cx = camera->cx_;
    const float &cy = camera->cy_;

    p2ds.clear();
    p2ds.reserve(p3ds.size());

    for(int i=0, iend=p3ds.size(); i<iend; i++)
    {
        Vector3d pcam = Scw * p3ds[i];
        const double invz = 1.0 / pcam[2];
        const double x = pcam[0] * invz;
        const double y = pcam[1] * invz;
        
        const float u = static_cast<float>(x) * fx + cx;
        const float v = static_cast<float>(y) * fy + cy;
        p2ds.push_back(Vector2d(u,v));
    }
}

int Sim3Solver::randomInt(int min, int max)
{
    int d = max - min + 1;
    return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
}

}
