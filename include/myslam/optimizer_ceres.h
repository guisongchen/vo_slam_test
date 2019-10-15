#ifndef OPTIMIZER_CERES_H
#define OPTIMIZER_CERES_H

#include "myslam/keyframe.h"
#include "myslam/loopClosing.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace myslam
{
    
class Optimizer
{
public:
    
    static int solvePoseOnlySE3(Frame* frame_curr);
    
    static void solveLocalBAPoseAndPoint(KeyFrame* keyframe, bool &stopFlag, Map* map_curr);
    
    static int solveLoopSim3(KeyFrame* keyframe_curr, KeyFrame* keyframe_match, 
                             vector<MapPoint*> &inlierMappoints, Sophus::Sim3 &Scm, const bool &fixScaleFlag);
    
    static int solvePoseGraphLoop(Map* map_curr, KeyFrame* keyframe_match, KeyFrame* keyframe_curr, 
                                  const LoopClosing::KeyFrameAndPose &uncorrectPose,
                                  const LoopClosing::KeyFrameAndPose &correctPose,
                                  const map<KeyFrame*, set<KeyFrame*> > &loopConnections,
                                  const bool &fixScaleFlag);
   
    template<typename T> inline
    static void se3TransPoint(const T se3[6], const T pt[3], T result[3])
    {
        const T upsilon[3] = {se3[0], se3[1], se3[2]};   
        
        const T& a0 = se3[3];
        const T& a1 = se3[4];
        const T& a2 = se3[5];
        const T theta2 = a0 * a0 + a1 * a1 + a2 * a2;
        
        if (theta2 > T(std::numeric_limits<double>::epsilon())) 
        {
            const T theta = sqrt(theta2);
            const T costheta = cos(theta);
            const T sintheta = sin(theta);
            const T theta_inverse = T(1.0) / theta;

            const T w[3] = { a0 * theta_inverse,
                             a1 * theta_inverse,
                             a2 * theta_inverse };

            const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                                    w[2] * pt[0] - w[0] * pt[2],
                                    w[0] * pt[1] - w[1] * pt[0] };
            const T tmp =
                (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

            result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
            result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
            result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
            
            const T w_cross_upsilon[3] = { w[1] * upsilon[2] - w[2] * upsilon[1],
                                           w[2] * upsilon[0] - w[0] * upsilon[2],
                                           w[0] * upsilon[1] - w[1] * upsilon[0] };
                                        
            const T w_double_cross_upsilon[3] = { w[1] * w_cross_upsilon[2] - w[2] * w_cross_upsilon[1],
                                                  w[2] * w_cross_upsilon[0] - w[0] * w_cross_upsilon[2],
                                                  w[0] * w_cross_upsilon[1] - w[1] * w_cross_upsilon[0] };
            
            result[0] += upsilon[0] + ((T(1.0)-costheta)/theta) * w_cross_upsilon[0] 
                        + ((theta-sintheta)/theta) * w_double_cross_upsilon[0];
            result[1] += upsilon[1] + ((T(1.0)-costheta)/theta) * w_cross_upsilon[1] 
                        + ((theta-sintheta)/theta) * w_double_cross_upsilon[1];
            result[2] += upsilon[2] + ((T(1.0)-costheta)/theta) * w_cross_upsilon[2] 
                        + ((theta-sintheta)/theta) * w_double_cross_upsilon[2];
            
            
        }
        else
        {
            const T w_cross_pt[3] = { a1 * pt[2] - a2 * pt[1],
                                      a2 * pt[0] - a0 * pt[2],
                                      a0 * pt[1] - a1 * pt[0] };

            result[0] = pt[0] + w_cross_pt[0];
            result[1] = pt[1] + w_cross_pt[1];
            result[2] = pt[2] + w_cross_pt[2];
            
            const T w_cross_upsilon[3] = { a1 * upsilon[2] - a2 * upsilon[1],
                                           a2 * upsilon[0] - a0 * upsilon[2],
                                           a0 * upsilon[1] - a1 * upsilon[0] };
                                        
            result[0] += upsilon[0] + w_cross_upsilon[0];
            result[1] += upsilon[1] + w_cross_upsilon[1];
            result[2] += upsilon[2] + w_cross_upsilon[2];
        }
    }

};

class IntrinsicProjectionUV : public ceres::SizedCostFunction<2, 3>
{
public:
    IntrinsicProjectionUV(const double* observation, const double* camera, const double& information);
    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const;
    
protected:
    const double  u_, v_;
    const double  fx_, fy_, cx_, cy_;
    const double  information_;
};

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 6; };
    virtual int LocalSize() const { return 6; };
};
    
class PoseOnlySE3UV : public ceres::SizedCostFunction<2, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PoseOnlySE3UV(const Vector3d &p3d, const Vector3d &pixel,
                const double* camera, const double &information);
    
    // NOTICE only one parameter block here, so jacobians will be only one row, cols = 2 * 6
    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const;

protected:
    Vector3d p_world_;
    const double u_, v_;
    const double fx_, fy_, cx_, cy_;
    const double information_;
};


class PoseOnlyStereoSE3UVD : public ceres::SizedCostFunction<3, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PoseOnlyStereoSE3UVD(const Vector3d &p3d, const Vector3d &pixel,
                         const double* camera, const double &information);
    
    // NOTICE only one parameter block here, so jacobians will be only one row, cols = 2 * 6
    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const;

protected:
    Vector3d p_world_;
    const double u_, v_, uR_;
    const double fx_, fy_, cx_, cy_, bf_;
    const double information_;
};
 

class LocalBAProjectUV : public ceres::SizedCostFunction<2, 6, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    LocalBAProjectUV(const Vector3d &pixel, const double* camera, const double &information);

    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const;

protected:
    const double u_, v_;
    const double fx_, fy_, cx_, cy_;
    const double information_;
    
};

class LocalBAStereoProjectUVD : public ceres::SizedCostFunction<3, 6, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    LocalBAStereoProjectUVD(const Vector3d &pixel, const double* camera, const double &information);

    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const;

protected:
    const double u_, v_, uR_;
    const double fx_, fy_, cx_, cy_, bf_;
    const double information_;
    
};

struct CameraProjection
{
    CameraProjection(const double* observation, const double* camera, const double& information)
        : intrinsicProjectionUV_(new IntrinsicProjectionUV(observation, camera, information)) {}
    
    template <typename T>
    bool operator () (const T* pose, const T* point, T* residuals) const
    {
        T pcam_estimate[3];
        
        ceres::AngleAxisRotatePoint(pose, point, pcam_estimate);
        pcam_estimate[0] += pose[3];
        pcam_estimate[1] += pose[4];
        pcam_estimate[2] += pose[5];
        
        return intrinsicProjectionUV_(pcam_estimate, residuals);
    }
    
private:
    ceres::CostFunctionToFunctor<2, 3> intrinsicProjectionUV_;
};

class PoseOnlySim3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PoseOnlySim3(const double* observation_curr, const double* camera, const double& information)
        : intrinsicProjectionUV_(new IntrinsicProjectionUV(observation_curr, camera, information)) {}
    
    template <typename T>
    bool operator () (const T* pose, const T* s_ptr, const T* point_cam_match, T* residuals) const
    {
        T Rcm_vec[9];
        T s = *s_ptr;
        ceres::AngleAxisToRotationMatrix(pose, Rcm_vec);
        
        Eigen::Map<const Eigen::Matrix<T, 3, 3> > Rcm(Rcm_vec);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > tcm(pose+3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pcam_match(point_cam_match);
        
        Eigen::Matrix<T, 3, 1> pcam_curr_estimate = s * Rcm * pcam_match + tcm;
   
        return intrinsicProjectionUV_(pcam_curr_estimate.data(), residuals);
    }

private:
    ceres::CostFunctionToFunctor<2, 3> intrinsicProjectionUV_;
};

class PoseOnlyInverseSim3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PoseOnlyInverseSim3(const double* observation_match, const double* camera, const double& information)
        : intrinsicProjectionUV_(new IntrinsicProjectionUV(observation_match, camera, information)) {}
    
    template <typename T>
    bool operator () (const T* pose, const T* s_ptr, const T* point_cam_curr, T* residuals) const
    {
        T Rcm_vec[9];
        T s = *s_ptr;
        ceres::AngleAxisToRotationMatrix(pose, Rcm_vec);
        
        Eigen::Map<const Eigen::Matrix<T, 3, 3> > Rcm(Rcm_vec);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > tcm(pose+3);
        Eigen::Matrix<T, 3, 3> sRmc = Rcm.transpose() / s;
        Eigen::Matrix<T, 3, 1> tmc = -sRmc * tcm;
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pcam_curr(point_cam_curr);
        Eigen::Matrix<T, 3, 1> pcam_match_estimate = sRmc * pcam_curr + tmc;
   
        return intrinsicProjectionUV_(pcam_match_estimate.data(), residuals);
    }

private:
    ceres::CostFunctionToFunctor<2, 3> intrinsicProjectionUV_;
};

class PoseGraphLoop
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // parameter values better to be stored in vector, or this will be really slow
    PoseGraphLoop(const Eigen::Quaterniond &q21_measured, const Vector3d &t21_measured, const double &s12_measured);
    
    
    // residuals: delta_theta(rotation), delta_translation, and delta_scale 
    // according to paper --strasdat-H 2012 PhD thesis
    template <typename T>
    bool operator () (const T* const q1_ptr, const T* const t1_ptr, const T* const s1_ptr,
                      const T* const q2_ptr, const T* const t2_ptr, const T* const s2_ptr,
                      T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Quaternion<T> > q1(q1_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > t1(t1_ptr);
        T s1 = *s1_ptr;

        Eigen::Map<const Eigen::Quaternion<T> > q2(q2_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > t2(t2_ptr);
        T s2 = *s2_ptr;
        
        Eigen::Map<Eigen::Matrix<T, 7, 1> > residuals(residuals_ptr);
        
        T s2_inv = T(1.0)/s2;
        Eigen::Quaternion<T> q2_inv = q2.conjugate();
        Eigen::Matrix<T, 3, 1> t2_inv = -(q2_inv * (s2_inv*t2));
        
        Eigen::Quaternion<T> q12_estimate = q1 * q2_inv;
        Eigen::Matrix<T, 3, 1> t12_estimate = s1*(q1*t2_inv) + t1;
        
        Eigen::Quaternion<T> delta_q = q21_measured_.template cast<T>() * q12_estimate;
        residuals.template block<3, 1>(0, 0) = T(2.0) * delta_q.vec();
        
        residuals.template block<3, 1>(3, 0) = s21_measured_ * (q21_measured_.template cast<T>()*t12_estimate)
                                               + t21_measured_.template cast<T>();
                                               
        residuals[6] = s21_measured_ * s1 * s2_inv;
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Quaterniond &q21_measured,
                                       const Vector3d &t21_measured,
                                       const double &s21_measured)
    {
        return (new ceres::AutoDiffCostFunction<PoseGraphLoop, 7, 4, 3, 1, 4, 3, 1>(
            new PoseGraphLoop(q21_measured, t21_measured, s21_measured)));
    }

private:
    Quaterniond     q21_measured_;
    Vector3d        t21_measured_;
    double          s21_measured_;
};


}

#endif
