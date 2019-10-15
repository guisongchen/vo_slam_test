#include "myslam/optimizer_ceres.h"
#include "myslam/mappoint.h"
#include <cmath>

namespace myslam
{

IntrinsicProjectionUV::IntrinsicProjectionUV(const double* observation,
                                             const double* camera,
                                             const double& information)
    : u_(observation[0]), v_(observation[1]), 
      fx_(camera[0]), fy_(camera[1]), cx_(camera[2]), cy_(camera[3]),
      information_(information) {}

bool IntrinsicProjectionUV::Evaluate(double const* const* parameter,
                                     double* residuals,
                                     double** jacobians) const
{
    const double* pcam = parameter[0];
    const double &x = pcam[0];
    const double &y = pcam[1];
    const double &z = pcam[2];
    const double invz = 1.0 / z;
    const double invz_2 = invz * invz;
    
    residuals[0] = (u_ - (fx_*x*invz + cx_)) * information_;   
    residuals[1] = (v_ - (fy_*y*invz + cy_)) * information_;
    
    if (!jacobians) return true;
    double* jacobian_uv_xyz = jacobians[0];
    if (!jacobian_uv_xyz) return true;
    
    jacobian_uv_xyz[0] = -invz * fx_;
    jacobian_uv_xyz[1] = 0;
    jacobian_uv_xyz[2] = x * invz_2 * fx_;
    
    jacobian_uv_xyz[3] = 0;
    jacobian_uv_xyz[4] = -invz * fy_;
    jacobian_uv_xyz[5] = y * invz_2 * fy_;
    
    return true;
}

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Matrix<double, 6, 1> > se3(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1> > update(delta);
    Eigen::Map<Eigen::Matrix<double, 6, 1> > se3PlusUpdate(x_plus_delta);
    se3PlusUpdate = (SE3::exp(update) * SE3::exp(se3)).log();
    
    return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > leftJacobian(jacobian);
    leftJacobian.setIdentity();
    
    return true;
}

PoseOnlySE3UV::PoseOnlySE3UV (const Vector3d &p3d,const Vector3d &pixel,
                              const double* camera, const double &information)
    : p_world_(p3d), u_(pixel[0]), v_(pixel[1]), fx_(camera[0]), fy_(camera[1]), cx_(camera[2]),
      cy_(camera[3]), information_(information) {}
    
bool PoseOnlySE3UV::Evaluate(double const* const* parameter, double* residuals, double** jacobians) const
{    
    double pcam[3];
    Optimizer::se3TransPoint(parameter[0], p_world_.data(), pcam);
    
    const double &x = pcam[0];
    const double &y = pcam[1];
    const double &z = pcam[2];
    const double invz = 1.0 / z;
    const double invz_2 = invz * invz;
    
    residuals[0] = (u_ - (fx_ * x * invz + cx_)) * information_;
    residuals[1] = (v_ - (fy_ * y * invz + cy_)) * information_;
    
    if (!jacobians) return true;
    
    double* jacobian_uv_kesai = jacobians[0];
    
    if (!jacobian_uv_kesai) return true;
    
    jacobian_uv_kesai[0] = -invz * fx_;
    jacobian_uv_kesai[1] = 0;
    jacobian_uv_kesai[2] = x * invz_2 * fx_;
    jacobian_uv_kesai[3] =  x * y * invz_2 * fx_;
    jacobian_uv_kesai[4] = -(1 + (x * x * invz_2)) * fx_;
    jacobian_uv_kesai[5] = y * invz * fx_;

    jacobian_uv_kesai[6] = 0;
    jacobian_uv_kesai[7] = -invz * fy_;
    jacobian_uv_kesai[8] = y * invz_2 * fy_;
    jacobian_uv_kesai[9] = (1 + y * y * invz_2) * fy_;
    jacobian_uv_kesai[10] = -x * y * invz_2 * fy_;
    jacobian_uv_kesai[11] = -x * invz * fy_;
    
    return true;
}

PoseOnlyStereoSE3UVD::PoseOnlyStereoSE3UVD (const Vector3d &p3d,const Vector3d &pixel,
                                            const double* camera, const double &information)
    : p_world_(p3d), u_(pixel[0]), v_(pixel[1]), uR_(pixel[2]), fx_(camera[0]), fy_(camera[1]),
      cx_(camera[2]), cy_(camera[3]), bf_(camera[4]), information_(information) {}
    
bool PoseOnlyStereoSE3UVD::Evaluate(double const* const* parameter, double* residuals, double** jacobians) const
{
    double pcam[3];
    Optimizer::se3TransPoint(parameter[0], p_world_.data(), pcam);
    
    const double &x = pcam[0];
    const double &y = pcam[1];
    const double &z = pcam[2];
    const double invz = 1.0 / z;
    const double invz_2 = invz * invz;
    const double uL_hat = fx_ * x * invz + cx_;
    
    residuals[0] = (u_ - uL_hat) * information_;
    residuals[1] = (v_ - (fy_ * y *invz + cy_)) * information_;
    residuals[2] = (uR_ - (uL_hat - bf_ * invz)) * information_;
    
    if (!jacobians) return true;
    
    double* jacobian_uvd_kesai = jacobians[0];
    
    if (!jacobian_uvd_kesai) return true;
    
    jacobian_uvd_kesai[0] = -invz * fx_;
    jacobian_uvd_kesai[1] = 0;
    jacobian_uvd_kesai[2] = x * invz_2 * fx_;
    jacobian_uvd_kesai[3] =  x * y * invz_2 * fx_;
    jacobian_uvd_kesai[4] = -(1 + (x * x * invz_2)) * fx_;
    jacobian_uvd_kesai[5] = y * invz * fx_;

    jacobian_uvd_kesai[6] = 0;
    jacobian_uvd_kesai[7] = -invz * fy_;
    jacobian_uvd_kesai[8] = y * invz_2 * fy_;
    jacobian_uvd_kesai[9] = (1 + y * y * invz_2) * fy_;
    jacobian_uvd_kesai[10] = -x * y * invz_2 * fy_;
    jacobian_uvd_kesai[11] = -x * invz * fy_;
  
    jacobian_uvd_kesai[12] = jacobian_uvd_kesai[0];
    jacobian_uvd_kesai[13] = 0;
    jacobian_uvd_kesai[14] = jacobian_uvd_kesai[2] - bf_ * invz_2;
    jacobian_uvd_kesai[15] = jacobian_uvd_kesai[3] - bf_ * y * invz_2;
    jacobian_uvd_kesai[16] = jacobian_uvd_kesai[4] + bf_ * x * invz_2;
    jacobian_uvd_kesai[17] = jacobian_uvd_kesai[5];

    
    return true;
}


int Optimizer::solvePoseOnlySE3 (Frame* frame_curr)
{
    int inlier_cnt = 0;    
    double pose[6];
    double pose_backup[6];
    
    Eigen::Matrix<double, 6, 1> se3 = frame_curr->Tcw_.log();
    memcpy(pose, se3.data(), sizeof(pose));
    memcpy(pose_backup, pose, sizeof(pose));
    
    Camera* camera_curr = frame_curr->camera_;
    const float &fx = camera_curr->fx_;
    const float &fy = camera_curr->fy_;
    const float &cx = camera_curr->cx_;
    const float &cy = camera_curr->cy_;
    const float &bf = camera_curr->bf_;
    const double camera[5] = {fx, fy, cx, cy, bf};
    
    vector<Vector3d> p3ds;
    vector<Vector3d> pixels;
    vector<int> indexs;
    vector<double> invSigmas;
    vector<double> invSigmas2;
    
    {
        unique_lock<mutex> lock(MapPoint::mutexOptimizer_);
        
        for (int i = 0, N = frame_curr->mappoints_.size(); i < N; i++)
        {
            MapPoint* mp = frame_curr->mappoints_[i];
            if (mp)
            {
                cv::KeyPoint kp = frame_curr->unKeypoints_[i];
                double invSigma = 1.0 / static_cast<double>(frame_curr->scaleFactors_[kp.octave]);
                
                p3ds.push_back(mp->getPose());
                pixels.push_back(Vector3d(kp.pt.x, kp.pt.y, frame_curr->uRight_[i]));
                
                indexs.push_back(i);
                invSigmas.push_back(invSigma);
                invSigmas2.push_back(invSigma*invSigma);
                
                frame_curr->outliers_[i] = false;
            }
        }
    }
    
    if (indexs.empty())
        return 0;
    
    SE3 Tcw;
    for (int iter = 0; iter < 2; iter++)
    {
        ceres::Problem problem;
        ceres::LocalParameterization* poseLocalParameter = new PoseLocalParameterization();
        ceres::LossFunction* lossFuncUV = static_cast<ceres::LossFunction*>(nullptr);
        ceres::LossFunction* lossFuncUVD = static_cast<ceres::LossFunction*>(nullptr);
        
        memcpy(pose, pose_backup, sizeof(pose_backup));
    
        if (iter < 1)
        {
            lossFuncUV = new ceres::HuberLoss(sqrt(5.991f));
            lossFuncUVD = new ceres::HuberLoss(sqrt(7.815f));
        }
        
        for (int i = 0; i < indexs.size(); i++)
        {
            int idx = indexs[i];
            
            if (!frame_curr->outliers_[idx])
            {
                Vector3d pix = pixels[i];
                
                if (pix[2] < 0)
                {
                    ceres::CostFunction* cost_function = new PoseOnlySE3UV(p3ds[i], pix, camera, invSigmas[i]);
                    problem.AddResidualBlock(cost_function, lossFuncUV, pose);  
                }
                else
                {
                    ceres::CostFunction* cost_function = new PoseOnlyStereoSE3UVD(p3ds[i], pix, 
                                                                                  camera, invSigmas[i]);
                    problem.AddResidualBlock(cost_function, lossFuncUVD, pose); 
                }
            }
        }
        
        problem.SetParameterization(pose, poseLocalParameter);
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 10;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
//         cout << summary.FullReport() << endl;
        
        Eigen::Map<const Eigen::Matrix<double, 6, 1> > se3(pose);
        Tcw = SE3::exp(se3);
        
        inlier_cnt = 0;
        for (int i = 0, N = indexs.size(); i < N; i++)
        {
            int idx = indexs[i];
            
            Vector3d pix = pixels[i];
            Vector3d pcam = Tcw * p3ds[i];
            
            const double &x = pcam[0];
            const double &y = pcam[1];
            const double &z = pcam[2];
            
            const float invz = 1.0f/z;
            const float u = fx*x*invz + cx;
            const float v = fy*y*invz + cy;
            
            const float eu = u - pix[0];
            const float ev = v - pix[1];
            const float e2 = eu*eu + ev*ev;
            const float invSigma2 = invSigmas2[i];
            
            if (pix[2] < 0)
            {
                if (e2*invSigma2 < 5.991f)
                {
                    frame_curr->outliers_[idx] = false;
                    inlier_cnt++;
                }
                else
                    frame_curr->outliers_[idx] = true;
            }
            else
            {
                const float ur = u - bf*invz; 
                const float e_ur = ur - pix[2];
                const float eu2 = e2 + e_ur*e_ur;
                
                if (eu2*invSigma2 < 7.815f)
                {
                    frame_curr->outliers_[idx] = false;
                    inlier_cnt++;
                }
                else
                    frame_curr->outliers_[idx] = true;
            }
        }
        
        if (inlier_cnt < 10)
            break;
    }
    
//     cout << "inlier num: " << inlier_cnt << endl;
    frame_curr->setPose(Tcw);
         
    return inlier_cnt;
}

LocalBAProjectUV::LocalBAProjectUV(const Vector3d &pixel, const double* camera, const double &information)
        : u_(pixel[0]), v_(pixel[1]), fx_(camera[0]), fy_(camera[1]),
          cx_(camera[2]), cy_(camera[3]), information_(information) {}

bool LocalBAProjectUV::Evaluate(double const* const* parameter, double* residuals, double** jacobians) const
{    
    const double* se3 = parameter[0];
    double pcam[3];
    Optimizer::se3TransPoint(se3, parameter[1], pcam);
    double R[9];
    const double r[3] = {se3[3], se3[4], se3[5]};
    ceres::AngleAxisToRotationMatrix(r, R);
    
    const double &x = pcam[0];
    const double &y = pcam[1];
    const double &z = pcam[2];
    const double invz = 1.0/z;
    const double invz_2 = invz*invz;
    
    residuals[0] = (u_ - (fx_*x*invz + cx_)) * information_;
    residuals[1] = (v_ - (fy_*y*invz + cy_)) * information_;
    
    if (!jacobians) return true;
    
    double* jacobian_uv_kesai = jacobians[0];
    double* jacobian_uv_xyz = jacobians[1];
    
    if (jacobian_uv_kesai)
    {
        jacobian_uv_kesai[0] = -invz * fx_;
        jacobian_uv_kesai[1] = 0;
        jacobian_uv_kesai[2] = x * invz_2 * fx_;
        jacobian_uv_kesai[3] =  x * y * invz_2 * fx_;
        jacobian_uv_kesai[4] = -(1 + (x * x * invz_2)) * fx_;
        jacobian_uv_kesai[5] = y * invz * fx_;

        jacobian_uv_kesai[6] = 0;
        jacobian_uv_kesai[7] = -invz * fy_;
        jacobian_uv_kesai[8] = y * invz_2 * fy_;
        jacobian_uv_kesai[9] = (1 + y * y * invz_2) * fy_;
        jacobian_uv_kesai[10] = -x * y * invz_2 * fy_;
        jacobian_uv_kesai[11] = -x * invz * fy_;
    }
    
    if (jacobian_uv_xyz)
    {
        jacobian_uv_xyz[0] = -fx_ * R[0] * invz + fx_ * x * R[2] * invz_2;
        jacobian_uv_xyz[1] = -fx_ * R[3] * invz + fx_ * x * R[5] * invz_2;
        jacobian_uv_xyz[2] = -fx_ * R[6] * invz + fx_ * x * R[8] * invz_2;

        jacobian_uv_xyz[3] = -fy_ * R[1] * invz + fy_ * y * R[2] * invz_2;
        jacobian_uv_xyz[4] = -fy_ * R[4] * invz + fy_ * y * R[5] * invz_2;
        jacobian_uv_xyz[5] = -fy_ * R[7] * invz + fy_ * y * R[8] * invz_2;
    }
    
    return true;
}

LocalBAStereoProjectUVD::LocalBAStereoProjectUVD(const Vector3d &pixel, const double* camera, 
                                                 const double &information)
                        : u_(pixel[0]), v_(pixel[1]), uR_(pixel[2]), fx_(camera[0]), fy_(camera[1]),
                          cx_(camera[2]), cy_(camera[3]), bf_(camera[4]), information_(information) {}

bool LocalBAStereoProjectUVD::Evaluate(double const* const* parameter, double* residuals, double** jacobians) const
{
    const double* se3 = parameter[0];
    double pcam[3];
    Optimizer::se3TransPoint(se3, parameter[1], pcam);
    double R[9];
    const double r[3] = {se3[3], se3[4], se3[5]};
    ceres::AngleAxisToRotationMatrix(r, R);
    
    const double &x = pcam[0];
    const double &y = pcam[1];
    const double &z = pcam[2];
    const double invz = 1.0 / z;
    const double invz_2 = invz * invz;
    const double uL_hat = fx_ * x * invz + cx_;
    
    residuals[0] = (u_ - uL_hat) * information_;
    residuals[1] = (v_ - (fy_ * y * invz + cy_)) * information_;
    residuals[2] = (uR_ - (uL_hat - bf_ * invz)) * information_;
    
    if (!jacobians) return true;
    
    double* jacobian_uvd_kesai = jacobians[0];
    double* jacobian_uvd_xyz = jacobians[1];
    
    if (jacobian_uvd_kesai)
    {
        jacobian_uvd_kesai[0] = -invz * fx_;
        jacobian_uvd_kesai[1] = 0;
        jacobian_uvd_kesai[2] = x * invz_2 * fx_;
        jacobian_uvd_kesai[3] =  x * y * invz_2 * fx_;
        jacobian_uvd_kesai[4] = -(1 + (x * x * invz_2)) * fx_;
        jacobian_uvd_kesai[5] = y * invz * fx_;

        jacobian_uvd_kesai[6] = 0;
        jacobian_uvd_kesai[7] = -invz * fy_;
        jacobian_uvd_kesai[8] = y * invz_2 * fy_;
        jacobian_uvd_kesai[9] = (1 + y * y * invz_2) * fy_;
        jacobian_uvd_kesai[10] = -x * y * invz_2 * fy_;
        jacobian_uvd_kesai[11] = -x * invz * fy_;
        
        jacobian_uvd_kesai[12] = jacobian_uvd_kesai[0];
        jacobian_uvd_kesai[13] = 0;
        jacobian_uvd_kesai[14] = jacobian_uvd_kesai[2] - bf_ * invz_2;
        jacobian_uvd_kesai[15] = jacobian_uvd_kesai[3] - bf_ * y * invz_2;
        jacobian_uvd_kesai[16] = jacobian_uvd_kesai[4] + bf_ * x * invz_2;
        jacobian_uvd_kesai[17] = jacobian_uvd_kesai[5];
    }
    
    if (jacobian_uvd_xyz)
    {
        jacobian_uvd_xyz[0] = -fx_ * R[0] * invz + fx_ * x * R[2] * invz_2;
        jacobian_uvd_xyz[1] = -fx_ * R[3] * invz + fx_ * x * R[5] * invz_2;
        jacobian_uvd_xyz[2] = -fx_ * R[6] * invz + fx_ * x * R[8] * invz_2;

        jacobian_uvd_xyz[3] = -fy_ * R[1] * invz + fy_ * y * R[2] * invz_2;
        jacobian_uvd_xyz[4] = -fy_ * R[4] * invz + fy_ * y * R[5] * invz_2;
        jacobian_uvd_xyz[5] = -fy_ * R[7] * invz + fy_ * y * R[8] * invz_2;

        jacobian_uvd_xyz[6] = jacobian_uvd_xyz[0] - bf_ * R[2] * invz_2;
        jacobian_uvd_xyz[7] = jacobian_uvd_xyz[1] - bf_ * R[5] * invz_2;
        jacobian_uvd_xyz[8] = jacobian_uvd_xyz[2] - bf_ * R[8] * invz_2;
    }
    
    return true;
}

void Optimizer::solveLocalBAPoseAndPoint(KeyFrame* keyframe, bool &stopFlag, Map* map_curr)
{
    
    vector<KeyFrame*> localKeyFrames;
    localKeyFrames.push_back(keyframe);
    keyframe->localBAKFId_ = keyframe->id_;
    
    const vector<KeyFrame*> neighbors = keyframe->getOrderedKFs();
    for (int i = 0, N = neighbors.size(); i < N; i++)
    {
        KeyFrame* kf = neighbors[i];
        
        //NOTE before bad juadgement
        kf->localBAKFId_ = keyframe->id_;
        
        if (!kf->isBad())
            localKeyFrames.push_back(kf);
    }
    
    vector<MapPoint*> localMapPoints;
    map<KeyFrame*, double*> observedKFPoses;
    vector<double*> localMapPointPositions;
    for (auto it = localKeyFrames.begin(), ite = localKeyFrames.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        
        SE3 Tcw = kf->getPose();
        double* pose = new double[6];
        Eigen::Matrix<double, 6, 1> se3 = Tcw.log();
        memcpy(pose, se3.data(), 6*sizeof(double));
        
        observedKFPoses.insert(make_pair(kf, pose));
        
        vector<MapPoint*> mappoints = kf->getMapPoints();
        for (int i = 0, N = mappoints.size(); i < N; i++)
        {
            MapPoint* mp = mappoints[i]; 
            if (!mp || mp->isBad())
                continue;
            
            if (mp->localBAKFId_ != keyframe->id_)
            {
                mp->localBAKFId_ = keyframe->id_;
                
                double* position = new double[3];
                const double* ptr = mp->getPose().data();
                position[0] = ptr[0];
                position[1] = ptr[1];
                position[2] = ptr[2];
                
                localMapPoints.push_back(mp);
                localMapPointPositions.push_back(position);
            }
        }
    }
    
    set<KeyFrame*> fixedKeyFrames;
    for (int i = 0, N = localMapPoints.size(); i < N; i++)
    {
        MapPoint* mp = localMapPoints[i];
        
        map<KeyFrame*, size_t> observations = mp->getObservedKFs();
        for (auto ito = observations.begin(), itoe = observations.end(); ito != itoe; ito++)
        {
            KeyFrame* kf = ito->first;
            
            if (kf->localBAKFId_ != keyframe->id_ && kf->BAFixId_ != keyframe->id_)
            {
                kf->BAFixId_ = keyframe->id_;
                
                if (!kf->isBad())
                {
                    SE3 Tcw = kf->getPose();
                    double* pose = new double[6];
                    Eigen::Matrix<double, 6, 1> se3 = Tcw.log();
                    memcpy(pose, se3.data(), 6*sizeof(double));
                    
                    observedKFPoses.insert(make_pair(kf, pose));
                    fixedKeyFrames.insert(kf);
                }
            }
        }
    }
    
    ceres::Problem problem1;
    ceres::LocalParameterization* poseLocalParameter = new PoseLocalParameterization();
    ceres::LossFunction* lossFuncUV = new ceres::HuberLoss(sqrt(5.991f));
    ceres::LossFunction* lossFuncUVD = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering1 = new ceres::ParameterBlockOrdering;
    
    vector<Vector3d> mapPointPixels;
    vector<double> invSigmas;
    
    vector< pair<KeyFrame*, MapPoint*> > edges;
    vector< pair<double*, double*> > poseAndPositions;
    
    Camera* camera_curr = keyframe->camera_;
    const float fx = camera_curr->fx_;
    const float fy = camera_curr->fy_;
    const float cx = camera_curr->cx_;
    const float cy = camera_curr->cy_;
    const float bf = camera_curr->bf_;
    const double camera[5] = {fx, fy, cx, cy, bf};
    
    for (int i = 0, N = localMapPoints.size(); i < N; i++)
    {
        MapPoint* mp = localMapPoints[i];        
        double* position = localMapPointPositions[i];
        
        const map<KeyFrame*, size_t> observations = mp->getObservedKFs();
        for (auto it = observations.begin(), ite = observations.end(); it != ite; it++)
        {
            KeyFrame* kf = it->first;
            const int idx = it->second;
            
            double* pose = observedKFPoses[kf];
            const cv::KeyPoint kpt = kf->unKeypoints_[idx];
            const float ur = kf->uRight_[idx];
            const double invSigma = 1.0 / static_cast<double>(kf->scaleFactors_[kpt.octave]);
            const Vector3d pixel(kpt.pt.x, kpt.pt.y, ur);
            
            if (ur < 0)
            {
                ceres::CostFunction* cost_function = new LocalBAProjectUV(pixel, camera, invSigma);
                problem1.AddResidualBlock(cost_function, lossFuncUV, pose, position);
            }
            else
            {
                ceres::CostFunction* cost_function = new LocalBAStereoProjectUVD(pixel, camera, invSigma);
                problem1.AddResidualBlock(cost_function, lossFuncUVD, pose, position);
            }
            
            if (fixedKeyFrames.count(kf) || kf->id_ == 0)
                problem1.SetParameterBlockConstant(pose);
            
            problem1.SetParameterization(pose, poseLocalParameter);
            
            ordering1->AddElementToGroup(position, 0);
            ordering1->AddElementToGroup(pose, 1);
            
            edges.push_back(make_pair(kf, mp));
            poseAndPositions.push_back(make_pair(pose, position));
            
            mapPointPixels.push_back(pixel);
            invSigmas.push_back(invSigma);
        }
    }
    
    if (stopFlag)
        return;

    ceres::Solver::Options options1;
    options1.linear_solver_type = ceres::DENSE_SCHUR;
    options1.linear_solver_ordering.reset(ordering1);
    options1.max_num_iterations = 5;
//     options1.num_threads = 2;
    
    ceres::Solver::Summary summary1;
    ceres::Solve(options1, &problem1, &summary1);
    
//     cout << summary1.FullReport() << endl;
    
    vector<bool> edgeOutliers(edges.size(), false);
    vector< pair<KeyFrame*, MapPoint*> > edgeErase;
    edgeErase.reserve(edges.size());
   
    if (!stopFlag)
    {
        ceres::Problem problem2;
        ceres::LocalParameterization* poseLocalParameter = new PoseLocalParameterization();
        ceres::ParameterBlockOrdering* ordering2 = new ceres::ParameterBlockOrdering;
        
        for (int i = 0, N = edges.size(); i < N; i++)
        {
            KeyFrame* kf = edges[i].first;
            
            Vector3d pixel = mapPointPixels[i];
            double* pose = poseAndPositions[i].first;
            double* position = poseAndPositions[i].second;
            
            double pcam[3];
            Optimizer::se3TransPoint(pose, position, pcam);
            
            const float x = pcam[0];
            const float y = pcam[1];
            const float z = pcam[2];
            
            if (z < 0.0f)
            {
                edgeOutliers[i] = true;
                continue;
            }
            
            const float invz = 1.0f/z;
            const float u = fx*x*invz + cx;
            const float v = fy*y*invz + cy;
            
            const float eu = u - static_cast<float>(pixel[0]);
            const float ev = v - static_cast<float>(pixel[1]);
            const float e2 = eu*eu + ev*ev;
            const float invSigma2 = invSigmas[i] * invSigmas[i];
            
            if (static_cast<float>(pixel[2]) < 0)
            {
                if (e2*invSigma2 > 5.991f)
                    edgeOutliers[i] = true;
                
                else
                {
                    ceres::CostFunction* cost_function = new LocalBAProjectUV(pixel, camera, invSigmas[i]);
                    problem2.AddResidualBlock(cost_function, nullptr, pose, position);
                    
                    if (fixedKeyFrames.count(kf) || kf->id_ == 0)
                        problem2.SetParameterBlockConstant(pose);
                    
                    ordering2->AddElementToGroup(position, 0);
                    ordering2->AddElementToGroup(pose, 1);
                    
                    problem2.SetParameterization(pose, poseLocalParameter);
                }
            }
            else
            {
                const float ur = u - bf*invz; 
                const float e_ur = ur - static_cast<float>(pixel[2]);
                const float eu2 = e2 + e_ur*e_ur;
                
                if (eu2*invSigma2 > 7.815f)
                    edgeOutliers[i] = true;
                else
                {
                    ceres::CostFunction* cost_function = new LocalBAStereoProjectUVD(pixel, camera, invSigmas[i]);
                    problem2.AddResidualBlock(cost_function, nullptr, pose, position);
                    
                    if (fixedKeyFrames.count(kf) || kf->id_ == 0)
                        problem2.SetParameterBlockConstant(pose);
                    
                    ordering2->AddElementToGroup(position, 0);
                    ordering2->AddElementToGroup(pose, 1);
                    
                    problem2.SetParameterization(pose, poseLocalParameter);
                }
            }
        }
        
        ceres::Solver::Options options2;
        options2.linear_solver_type = ceres::DENSE_SCHUR;
        options2.linear_solver_ordering.reset(ordering2);
        options2.max_num_iterations = 10;
//         options2.num_threads = 2;
//         options.minimizer_progress_to_stdout = true;
        
        ceres::Solver::Summary summary2;
        ceres::Solve(options2, &problem2, &summary2);
    }
    
    
    for (int i = 0, N = edges.size(); i < N; i++)
    {
        
        KeyFrame* kf = edges[i].first;
        MapPoint* mp = edges[i].second;
        
        if (edgeOutliers[i])
        {
            edgeErase.push_back(make_pair(kf, mp));
            continue;
        }
        
        Vector3d pixel = mapPointPixels[i];
        double* pose = poseAndPositions[i].first;
        double* position = poseAndPositions[i].second;
        
        double pcam[3];
        Optimizer::se3TransPoint(pose, position, pcam);
        
        const float x = pcam[0];
        const float y = pcam[1];
        const float z = pcam[2];
        
        if (z < 0.0f)
        {
            edgeErase.push_back(make_pair(kf, mp));
            continue;
        }
        
        const float invz = 1.0f/z;
        const float u = fx*x*invz + cx;
        const float v = fy*y*invz + cy;
        
        const float eu = u - pixel[0];
        const float ev = v - pixel[1];
        const float e2 = eu*eu + ev*ev;
        const float invSigma2 = invSigmas[i] * invSigmas[i];
        
        if (pixel[2] < 0)
        {
            if (e2*invSigma2 > 5.991f)
                edgeErase.push_back(make_pair(kf, mp));
        }
        else
        {
            const float ur = u - bf*invz; 
            const float e_ur = ur - pixel[2];
            const float eu2 = e2 + e_ur*e_ur;
            
            if (eu2*invSigma2 > 7.815f)
                edgeErase.push_back(make_pair(kf, mp));
        }
    }

    {
        unique_lock<mutex> lock(map_curr->mutexMapUpdate_);
        
        if (!edgeErase.empty())
        {
            for (int i = 0, N = edgeErase.size(); i < N; i++)
            {
                KeyFrame* kf = edgeErase[i].first;
                MapPoint* mp = edgeErase[i].second;
                
                int idx = mp->getIndexInKeyFrame(kf);
                if (idx > 0)
                    kf->setMapPointNull(idx);
                
                mp->eraseObservedKF(kf);
            }
        }
        
        for (auto it = observedKFPoses.begin(), ite = observedKFPoses.end(); it != ite; it++)
        {
            KeyFrame* kf = it->first;
            double* pose = it->second;
            
            if (fixedKeyFrames.count(kf) || kf->id_ == 0)
            {
                delete[] pose;
                continue;
            }
            
            Eigen::Map<const Eigen::Matrix<double, 6, 1> > se3(pose);
            SE3 Tcw = SE3::exp(se3);
            kf->setPose(Tcw);

            delete[] pose;
        }
        
        for (int i = 0, N = localMapPoints.size(); i < N; i++)
        {
            MapPoint* mp = localMapPoints[i];
            double* position = localMapPointPositions[i];
            
            Eigen::Map<const Vector3d> pos(position);
            mp->setPose(pos);
            mp->updateNormalAndDepth();
            
            delete[] position;
        }
    }
    
    observedKFPoses.clear();
    localMapPointPositions.clear();
}

int Optimizer::solveLoopSim3(KeyFrame* keyframe_curr, KeyFrame* keyframe_match, 
                             vector<MapPoint*> &inlierMappoints, Sophus::Sim3 &Scm, const bool &fixScaleFlag)
{
    cout << "start sim3 optimizing..." << endl;
    
    SE3 Tcw = keyframe_curr->getPose();
    SE3 Tmw = keyframe_match->getPose();
    
    vector<MapPoint*> mappoints1 = keyframe_curr->getMapPoints();

    double pose[6];
    double s = Scm.scale();

    // Sim3 output rotation_matrix and translation without scale
    Matrix3d Rcm = Scm.rotation_matrix();
    Vector3d tcm = Scm.translation();
    ceres::RotationMatrixToAngleAxis(Rcm.data(), pose);
    memcpy(pose+3, tcm.data(), 3*sizeof(double));
    
    Camera* cameraPtr = keyframe_curr->camera_;
    const float fx = cameraPtr->fx_;
    const float fy = cameraPtr->fy_;
    const float cx = cameraPtr->cx_;
    const float cy = cameraPtr->cy_;
    const double camera[4] = {fx, fy, cx, cy};
         
    vector<Vector3d> cameraCurr;
    vector<Vector2d> pixelCurr;
    vector<double> invSigmaCurr;
    vector<Vector3d> cameraMatch;
    vector<Vector2d> pixelMatch;
    vector<double> invSigmaMatch;
    
    int match_cnt = 0;
    vector<int> indexs;
    indexs.reserve(inlierMappoints.size());
    
    for (int i = 0, N = inlierMappoints.size(); i < N; i++)
    {
        MapPoint* mpm = inlierMappoints[i];
        if (!mpm || mpm->isBad())
            continue;
        
        MapPoint* mpc = mappoints1[i];
        if (!mpc || mpc->isBad())
            continue;
        
        int idx_match = mpm->getIndexInKeyFrame(keyframe_match);
        if (idx_match < 0)
            continue;
        
        Vector3d cam_match = Tmw * mpm->getPose();
        cv::KeyPoint kpt_m = keyframe_match->unKeypoints_[idx_match];
        Vector2d pxl_match(kpt_m.pt.x, kpt_m.pt.y);
        double invSigma_m = 1.0 / static_cast<double>(keyframe_match->scaleFactors_[kpt_m.octave]);
        
        cameraMatch.push_back(cam_match);
        pixelMatch.push_back(pxl_match);
        invSigmaMatch.push_back(invSigma_m);
        
        Vector3d cam_curr = Tcw *  mpc->getPose();
        cv::KeyPoint kpt_c = keyframe_curr->unKeypoints_[i];
        Vector2d pxl_curr(kpt_c.pt.x, kpt_c.pt.y);
        double invSigma_c = 1.0 / static_cast<double>(keyframe_curr->scaleFactors_[kpt_c.octave]);
        
        cameraCurr.push_back(cam_curr);
        pixelCurr.push_back(pxl_curr);
        invSigmaCurr.push_back(invSigma_c);
        
        indexs.push_back(i);
        match_cnt++;
    }

    ceres::Problem problem1;
    ceres::LossFunction* lossFunc1 = new ceres::HuberLoss(sqrt(10.0f));
    
    for (int i = 0; i < match_cnt; i++)
    {
        ceres::CostFunction* cost_function1 = new ceres::AutoDiffCostFunction<PoseOnlySim3, 2, 6, 1, 3>(
            new PoseOnlySim3(pixelCurr[i].data(), camera, invSigmaCurr[i]));
        
        ceres::CostFunction* cost_function2 = new ceres::AutoDiffCostFunction<PoseOnlyInverseSim3, 2, 6, 1, 3>(
            new PoseOnlyInverseSim3(pixelMatch[i].data(), camera, invSigmaMatch[i]));
        
        problem1.AddResidualBlock(cost_function1, lossFunc1, pose, &s, cameraMatch[i].data());
        problem1.AddResidualBlock(cost_function2, lossFunc1, pose, &s, cameraCurr[i].data());
        
        problem1.SetParameterBlockConstant(cameraMatch[i].data());
        problem1.SetParameterBlockConstant(cameraCurr[i].data());
    }
    
    if (fixScaleFlag)
        problem1.SetParameterBlockConstant(&s);
    
    ceres::Solver::Options options1;
    options1.linear_solver_type = ceres::DENSE_SCHUR;
    options1.max_num_iterations = 10;
    
    ceres::Solver::Summary summary1;
    ceres::Solve(options1, &problem1, &summary1);
    
    double Rcm1_array[9];
    ceres::AngleAxisToRotationMatrix(pose, Rcm1_array);
    Eigen::Map<const Eigen::Matrix3d> Rcm1(Rcm1_array);
    Eigen::Map<const Vector3d> tcm1(pose+3);
    
    Sophus::Sim3 Scm1(Sophus::ScSO3(s, Rcm1), tcm1);
    Sophus::Sim3 Smc1 = Scm1.inverse();
    
    int outlier_cnt = 0;
    vector<bool> outlierFlags(match_cnt, false);
    for (int i = 0; i < match_cnt; i++)
    {
        Vector3d cam_curr_est = Scm1 * cameraMatch[i];        
        Vector2d pix_curr_est = cameraPtr->camera2pixel(cam_curr_est);
        const double e_curr = (pix_curr_est - pixelCurr[i]).squaredNorm();
        
        if (e_curr*invSigmaCurr[i]*invSigmaCurr[i] > 10.0)
        {
            int idx = indexs[i];
            inlierMappoints[idx] = static_cast<MapPoint*>(nullptr);
            
            outlierFlags[i] = true;
            outlier_cnt++;
            continue;
        }
        
        Vector3d cam_match_est = Smc1 * cameraCurr[i];
        Vector2d pix_match_est = cameraPtr->camera2pixel(cam_match_est);
        const double e_match = (pix_match_est - pixelMatch[i]).squaredNorm();
        
        if (e_match*invSigmaMatch[i]*invSigmaMatch[i] > 10.0)
        {
            int idx = indexs[i];
            inlierMappoints[idx] = static_cast<MapPoint*>(nullptr);
            
            outlierFlags[i] = true;
            outlier_cnt++;
            continue;
        }
    }
    
    if (match_cnt - outlier_cnt < 10)
        return 0;
    
    int moreIteration;
    if (outlier_cnt > 0)
        moreIteration = 10;
    else
        moreIteration = 5;
    
    ceres::Problem problem2;
    ceres::LossFunction* lossFunc2 = new ceres::HuberLoss(sqrt(10.0f));
    for (int i = 0; i < match_cnt; i++)
    {
        if (outlierFlags[i])
            continue;
        
        ceres::CostFunction* cost_function1 = new ceres::AutoDiffCostFunction<PoseOnlySim3, 2, 6, 1, 3>(
            new PoseOnlySim3(pixelCurr[i].data(), camera, invSigmaCurr[i]));
        
        ceres::CostFunction* cost_function2 = new ceres::AutoDiffCostFunction<PoseOnlyInverseSim3, 2, 6, 1, 3>(
            new PoseOnlyInverseSim3(pixelMatch[i].data(), camera, invSigmaMatch[i]));
        
        problem2.AddResidualBlock(cost_function1, lossFunc2, pose, &s, cameraMatch[i].data());
        problem2.AddResidualBlock(cost_function2, lossFunc2, pose, &s, cameraCurr[i].data());
        
        problem2.SetParameterBlockConstant(cameraMatch[i].data());
        problem2.SetParameterBlockConstant(cameraCurr[i].data());
    }
    
    if (fixScaleFlag)
        problem2.SetParameterBlockConstant(&s);
    
    ceres::Solver::Options options2;
    options2.linear_solver_type = ceres::DENSE_SCHUR;
    options2.max_num_iterations = moreIteration;
    
    ceres::Solver::Summary summary2;
    ceres::Solve(options2, &problem2, &summary2);
    
    double Rcm2_array[9];
    ceres::AngleAxisToRotationMatrix(pose, Rcm2_array);
    Eigen::Map<const Eigen::Matrix3d> Rcm2(Rcm2_array);
    Eigen::Map<const Vector3d> tcm2(pose+3);
    
    Sophus::Sim3 Scm2(Sophus::ScSO3(s, Rcm2), tcm2);
    Sophus::Sim3 Smc2 = Scm2.inverse();
    
    int inlier_cnt = 0;
    for (int i = 0; i < match_cnt; i++)
    {
        Vector3d cam_curr_est = Scm2 * cameraMatch[i];        
        Vector2d pix_curr_est = cameraPtr->camera2pixel(cam_curr_est);
        const double e_curr = (pix_curr_est - pixelCurr[i]).squaredNorm();
        
        if (e_curr*invSigmaCurr[i]*invSigmaCurr[i] > 10.0)
        {
            int idx = indexs[i];
            inlierMappoints[idx] = static_cast<MapPoint*>(nullptr);
            continue;
        }
        
        Vector3d cam_match_est = Smc2 * cameraCurr[i];
        Vector2d pix_match_est = cameraPtr->camera2pixel(cam_match_est);
        const double e_match = (pix_match_est - pixelMatch[i]).squaredNorm();
        
        if (e_match*invSigmaMatch[i]*invSigmaMatch[i] > 10.0)
        {
            int idx = indexs[i];
            inlierMappoints[idx] = static_cast<MapPoint*>(nullptr);
            continue;
        }
        
        inlier_cnt++;
    }
    
    Scm = Scm2;
    
    return inlier_cnt;
}


PoseGraphLoop::PoseGraphLoop(const Eigen::Quaterniond &q21_measured, const Vector3d &t21_measured, const double &s21_measured)
    : q21_measured_(q21_measured), t21_measured_(t21_measured), s21_measured_(s21_measured) {}
    
int Optimizer::solvePoseGraphLoop(Map* map_curr, KeyFrame* keyframe_match, KeyFrame* keyframe_curr, 
                                  const LoopClosing::KeyFrameAndPose &uncorrectPose,
                                  const LoopClosing::KeyFrameAndPose &correctPose,
                                  const map<KeyFrame*, set<KeyFrame*> > &loopConnections,
                                  const bool &fixScaleFlag)
{
    cout << "start posegraph optimizing..." << endl;
    
    vector<KeyFrame*> allKeyFrames = map_curr->getAllKeyFrames();
    vector<MapPoint*> allMapPoints = map_curr->getAllMapPoints();
    
    unsigned long maxKFId = map_curr->maxKFId_;
    vector<Sophus::Sim3, Eigen::aligned_allocator<Sophus::Sim3> > Scw(maxKFId + 1);
    vector<Sophus::Sim3, Eigen::aligned_allocator<Sophus::Sim3> > optimizedSwc(maxKFId + 1);
    
    vector<Quaterniond> q_measures;
    vector<Vector3d> t_measures;
    vector<double> s_measures;
    
    vector<Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > uquats(maxKFId + 1);  // unit quaternion
    vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > trans(maxKFId + 1); // translation(unscaled)
    vector<double> scales(maxKFId + 1);
    
    list<unsigned long> KFIndexes;
    
    // collect keyframe pose, use corrected Sim3 if exist, wrap current Tcw to Sim3 otherwise
    for (size_t i = 0, N = allKeyFrames.size(); i < N; i++)
    {
        KeyFrame* kf = allKeyFrames[i];
        
        const unsigned long idx = kf->id_;
        LoopClosing::KeyFrameAndPose::const_iterator it = correctPose.find(kf);
        
        if (it != correctPose.end())
        {
            Scw[idx] = it->second;
        }
        else
        {
            SE3 Tiw = kf->getPose();
            Sophus::Sim3 Siw(Sophus::ScSO3(Tiw.unit_quaternion()), Tiw.translation());
            Scw[idx] = Siw;
        }
        
        uquats[idx] = Scw[idx].quaternion().normalized();
        trans[idx] = Scw[idx].translation();
        scales[idx] = Scw[idx].scale();
        
        KFIndexes.push_back(idx);
    }

    const int minFeat = 100;
    
    ceres::Problem problem;
    ceres::LocalParameterization* quatLocalParam = new ceres::EigenQuaternionParameterization;
    
    // set loopEdges using correctedPos
    set<pair<unsigned long, unsigned long> > insertedLoopEdges;
    for (auto it = loopConnections.begin(), ite = loopConnections.end(); it != ite; it++)
    {
        KeyFrame* kf = it->first;
        const unsigned long id1 = kf->id_;
        const set<KeyFrame*> &connections = it->second;
        Sophus::Sim3 &Siw = Scw[id1];
        Sophus::Sim3 Swi = Siw.inverse();
        
        for (auto itc = connections.begin(), itce = connections.end(); itc != itce; itc++)
        {
            const unsigned long id2 = (*itc)->id_;
            if ((id2!=keyframe_curr->id_ || id2!=keyframe_match->id_) && kf->getWeight(*itc)<minFeat)
                continue;
            
            Sophus::Sim3 &Sjw = Scw[id2];
            Sophus::Sim3 Sji = Sjw * Swi;
            
            q_measures.push_back(Sji.quaternion().normalized());
            t_measures.push_back(Sji.translation());
            s_measures.push_back(Sji.scale());
            
            ceres::CostFunction* cost_function = PoseGraphLoop::Create(q_measures.back(), 
                                                                       t_measures.back(),
                                                                       s_measures.back());
            
            problem.AddResidualBlock(cost_function, nullptr,
                                     uquats[id1].coeffs().data(), trans[id1].data(), &scales[id1],
                                     uquats[id2].coeffs().data(), trans[id2].data(), &scales[id2]);
            
            insertedLoopEdges.insert(make_pair(min(id1, id2), max(id1, id2)));
        }
    }
  
    // set expanding tree and good covsibles using uncorrectPose
    for (size_t i = 0, N = allKeyFrames.size(); i < N; i++)
    {
        KeyFrame* kf = allKeyFrames[i];
        
        const unsigned long id1 = kf->id_;
        LoopClosing::KeyFrameAndPose::const_iterator it1 = uncorrectPose.find(kf);
        Sophus::Sim3 Swi;
        
        if (it1 != uncorrectPose.end())
            Swi = it1->second.inverse();
        else
            Swi = Scw[id1].inverse();
        
        KeyFrame* parentKF = kf->getParent();
        if (parentKF)
        {
            const unsigned long id2 = parentKF->id_;
            LoopClosing::KeyFrameAndPose::const_iterator it2 = uncorrectPose.find(parentKF);
            Sophus::Sim3 Sjw;
            
            if (it2 != uncorrectPose.end())
                Sjw = it2->second;
            else
                Sjw = Scw[id2];
            
            Sophus::Sim3 Sji = Sjw * Swi;
            
            q_measures.push_back(Sji.quaternion().normalized());
            t_measures.push_back(Sji.translation());
            s_measures.push_back(Sji.scale());
            
            ceres::CostFunction* cost_function = PoseGraphLoop::Create(q_measures.back(), 
                                                                       t_measures.back(),
                                                                       s_measures.back());
            problem.AddResidualBlock(cost_function, nullptr,
                                     uquats[id1].coeffs().data(), trans[id1].data(), &scales[id1],
                                     uquats[id2].coeffs().data(), trans[id2].data(), &scales[id2]);
            
        }
        
        const set<KeyFrame*> loopEdges = kf->loopEdges_;
        for (auto itl = loopEdges.begin(), itle = loopEdges.end(); itl != itle; itl++)
        {
            KeyFrame* kfl = *itl;
            
            // keyframe before loop detected
            if (kfl->id_ < keyframe_curr->id_)
            {
                const unsigned long id2 = kfl->id_;
                
                Sophus::Sim3 Slw;
                LoopClosing::KeyFrameAndPose::const_iterator it2 = uncorrectPose.find(kfl);
                
                if (it2 != uncorrectPose.end())
                    Slw = it2->second;
                else
                    Slw = Scw[id2];
                
                Sophus::Sim3 Sli = Slw * Swi;
                q_measures.push_back(Sli.quaternion().normalized());
                t_measures.push_back(Sli.translation());
                s_measures.push_back(Sli.scale());
                
                ceres::CostFunction* cost_function = PoseGraphLoop::Create(q_measures.back(), 
                                                                           t_measures.back(),
                                                                           s_measures.back());
                problem.AddResidualBlock(cost_function, nullptr,
                                         uquats[id1].coeffs().data(), trans[id1].data(), &scales[id1],
                                         uquats[id2].coeffs().data(), trans[id2].data(), &scales[id2]);
            }
        }
        
        const vector<KeyFrame*> weightedCovisibleKFs = kf->getCovisiblesByWeight(minFeat);
        for (auto itw = weightedCovisibleKFs.begin(), itwe = weightedCovisibleKFs.end(); itw != itwe; itw++)
        {
            KeyFrame* kfw = *itw;
            if (kfw && kfw != parentKF && !kf->children_.count(kfw) && !loopEdges.count(kfw))
            {
                if (!kfw->isBad() && kfw->id_ < kf->id_)
                {
                    if (insertedLoopEdges.count(make_pair(kfw->id_, kf->id_)))
                        continue;
                    
                    const unsigned long id2 = kfw->id_;
                    Sophus::Sim3 Snw;
                    
                    LoopClosing::KeyFrameAndPose::const_iterator it2 = uncorrectPose.find(kfw);
                    
                    if (it2 != uncorrectPose.end())
                    {
                        Snw = it2->second;
                    }
                    else
                        Snw = Scw[id2];
                    
                    Sophus::Sim3 Sni = Snw * Swi;
                    q_measures.push_back(Sni.quaternion().normalized());
                    t_measures.push_back(Sni.translation());
                    s_measures.push_back(Sni.scale());
                    
                    ceres::CostFunction* cost_function = PoseGraphLoop::Create(q_measures.back(), 
                                                                               t_measures.back(),
                                                                               s_measures.back());
                    problem.AddResidualBlock(cost_function, nullptr,
                                             uquats[id1].coeffs().data(), trans[id1].data(), &scales[id1],
                                             uquats[id2].coeffs().data(), trans[id2].data(), &scales[id2]);
                }
            }
        }
    }

    problem.SetParameterBlockConstant(uquats[keyframe_match->id_].coeffs().data());
    problem.SetParameterBlockConstant(trans[keyframe_match->id_].data());
    problem.SetParameterBlockConstant(&scales[keyframe_match->id_]);
    
    for (auto idx:KFIndexes)
            problem.SetParameterization(uquats[idx].coeffs().data(), quatLocalParam);
    
    if (fixScaleFlag)
    {
        for (auto idx:KFIndexes)
            problem.SetParameterBlockConstant(&scales[idx]);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    options.num_threads = 2;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
//     cout << summary.FullReport() << endl;
    
    {
        unique_lock<mutex> lock(map_curr->mutexMapUpdate_);

        for (auto it = allKeyFrames.begin(), ite = allKeyFrames.end(); it != ite; it++)
        {
            KeyFrame* kf = *it;
            const unsigned long id = kf->id_;
            
            Quaterniond &uq = uquats[id];
            Vector3d &t = trans[id];
            double &s = scales[id];
            
            SE3 Tiw(uq, t/s);
            kf->setPose(Tiw);
            
            Sophus::Sim3 Siw(Sophus::ScSO3(s, Tiw.rotation_matrix()), t);
            optimizedSwc[id] = Siw.inverse();
        }
        
        for (size_t i = 0, N = allMapPoints.size(); i < N; i++)
        {
            MapPoint* mp = allMapPoints[i];
            if (mp->isBad())
                continue;
            
            unsigned long idm;
            if (mp->loopCorrectByKF_ == keyframe_curr->id_)
                idm = mp->correctReference_;
            else
                idm = mp->keyFrame_ref_->id_;
            
            Sophus::Sim3 Srw = Scw[idm];
            Sophus::Sim3 Swr = optimizedSwc[idm];
            
            Vector3d p_world = mp->getPose();
            Vector3d correctedPos = Swr * (Srw * p_world);
            mp->setPose(correctedPos);
            
            mp->updateNormalAndDepth();
        }
    }
    
    cout << "posegraph optimzed!!" << endl;
}
    
}
