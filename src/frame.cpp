#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace myslam
{

static long unsigned int factory_id = 0;

Frame::Frame(Mat &grayImg, Mat &depthImg, string timeStamp, Camera* camera, ORB_SLAM2::ORBextractor* orb)
     : timeStamp_(timeStamp), camera_(camera), Tcw_(SE3()), poseExist_(false), keyframe_trackRef_(nullptr),
       xMin_(camera->xMin_), xMax_(camera->xMax_), yMin_(camera->yMin_), yMax_(camera->yMax_),
       gridPerPixelWidth_(camera->gridPerPixelWidth_),
       gridPerPixelHeight_(camera->gridPerPixelHeight_),
       voc_(nullptr), orb_(orb)
{
    id_ = factory_id++;  

    (*orb_)(grayImg, Mat(), keypoints_, descriptors_);
    scaleFactors_ = orb_->GetScaleFactors();
    N_ = keypoints_.size();

    if(keypoints_.empty())
        return;
    
    undistortKeyPoints();
    findDepth(depthImg);
    assignFeaturesToGrid();
    mappoints_ = vector<MapPoint* >(N_, static_cast<MapPoint*>(nullptr));
    outliers_ = vector<bool>(N_, false);
}

void Frame::undistortKeyPoints()
{
    Mat distCoef = camera_->distCoef_;
    Mat K = camera_->K_;
    
    if (distCoef.at<float>(0) == 0.0f)
    {
        unKeypoints_ = keypoints_;
        return;
    }
    
    Mat mat(N_, 2, CV_32F);
    for (int i = 0; i < N_; i++)
    {
        mat.at<float>(i,0) = keypoints_[i].pt.x;
        mat.at<float>(i,1) = keypoints_[i].pt.y;
    }
    
    // N*2*1 -> N*1*2
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, distCoef, Mat(), K);
    
    // N*1*2 -> N*2*1
    mat = mat.reshape(1);
    
    unKeypoints_.resize(N_);
    
    for (int i = 0; i < N_; i++)
    {
        cv::KeyPoint kp = keypoints_[i];
        kp.pt.x = mat.at<float>(i,0);
        kp.pt.y = mat.at<float>(i,1);
        unKeypoints_[i] = kp;
    }
}

void Frame::assignFeaturesToGrid()
{
    int reserveNum = 0.5f * N_ / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            gridKeypoints_[i][j].reserve(reserveNum);
        
    for (int i = 0; i < N_; i++)
    {
        const cv::KeyPoint kpt = unKeypoints_[i];
        
        const int gradNumX = round((kpt.pt.x - xMin_) * gridPerPixelWidth_);
        const int gradNumY = round((kpt.pt.y - yMin_) * gridPerPixelHeight_);
        
        if (postionInGrad(gradNumX, gradNumY))
            gridKeypoints_[gradNumX][gradNumY].push_back(i);
    }
}

bool Frame::postionInGrad(const int &gradNumX, const int &gradNumY)
{
    if(gradNumX<0 || gradNumX>=FRAME_GRID_COLS || gradNumY<0 || gradNumY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::setPose(SE3 Tcw)
{
    Tcw_ = Tcw;
    Ow_ = Tcw_.inverse().translation();
    poseExist_ = true;
}


void Frame::findDepth(Mat &depthImg)
{
    if (keypoints_.empty())
        return;

    uRight_ = vector<float>(N_, -1);
    depth_ = vector<float>(N_, -1);
    
    const float bf = camera_->bf_;
    
    for (int i = 0; i < N_; i++)
    {
        const cv::KeyPoint &kp = keypoints_[i];
        const cv::KeyPoint &kpU = unKeypoints_[i];
        const float u = kp.pt.x;
        const float v = kp.pt.y;
        const float d = depthImg.at<float>(v,u);
        
        // depthImg can't be undistorted, so use original keypoint find depth
        if (d > 0)
        {
            depth_[i] = d;
            uRight_[i] = kpU.pt.x - bf/d;
        }
    }
}

vector<Mat> Frame::getDespVector()
{
    vector<Mat> desps;
    desps.reserve(N_);
    for (int i = 0; i < descriptors_.rows; i++)
        desps.push_back(descriptors_.row(i).clone());
    
    return desps;
}

bool Frame::isInFrame(MapPoint* mp)
{
    mp->trackInLocalMap_ = false;
    
    Vector3d p_world = mp->getPose();
    Vector3d p_cam = Tcw_ * p_world;
    const float z = static_cast<float>(p_cam[2]);
    
    if (z < 0.0f)
        return false;
    
    Vector2d pixel = camera_->camera2pixel(p_cam);
    
    const float u = pixel[0];
    if (u < xMin_ || u > xMax_)
        return false;
    
    const float v = pixel[1];
    if (v < yMin_ || v > yMax_)
        return false;
    
    const Vector3d line = p_world - Ow_;
    const float dist = line.norm();
    
    const float minDistance = mp->getMinDistanceThreshold();
    const float maxDistance = mp->getMaxDistanceThreshold();
    
    if ( dist < minDistance || dist > maxDistance)
        return false;
    
    const Vector3d normalVector = mp->getNormalVector();
    const float viewcos = static_cast<float>(line.dot(normalVector)) / dist;
    if (viewcos < 0.5f)
        return false;
    
    // information for search local mappoints
    const float bf = camera_->bf_;
    mp->trackInLocalMap_ = true;
    mp->trackProj_u_ = u;
    mp->trackProj_uR_ = u - bf/z; 
    mp->trackProj_v_ = v;
    mp->trackScaleLevel_ = mp->predictScale(dist, this);
    mp->viewCos_ = viewcos;
    
    return true;
}

bool Frame::isInImg(const Vector2d &pixel)
{
    const float &u = pixel[0];
    const float &v = pixel[1];
    return (u>xMin_ && u<xMax_ && v>yMin_ && v<yMax_);
}

vector<int> Frame::getFeaturesInArea(const float& u, const float& v,const float& radius,
                                     const int min_level, const int max_level)
{
    vector<int> indexs;
    indexs.reserve(N_);
    
    const int minGridNumX = max(0, static_cast<int>(floor((u-xMin_-radius)*gridPerPixelWidth_)));
    if (minGridNumX >= FRAME_GRID_COLS)
        return indexs;
    
    const int maxGridNumX = min(static_cast<int>(FRAME_GRID_COLS - 1),
                                static_cast<int>(floor((u-xMin_+radius)*gridPerPixelWidth_)));
    if (maxGridNumX < 0)
        return indexs;
    
    const int minGridNumY = max(0, static_cast<int>(floor((v-yMin_-radius)*gridPerPixelHeight_)));
    if (minGridNumY >= FRAME_GRID_ROWS)
        return indexs;
    
    const int maxGridNumY = min(static_cast<int>(FRAME_GRID_ROWS - 1),
                                static_cast<int>(floor((v-yMin_+radius)*gridPerPixelHeight_)));
    if (maxGridNumY < 0)
        return indexs;
    
    for (int ix = minGridNumX; ix <= maxGridNumX; ix++)
    {
        for (int iy = minGridNumY; iy <= maxGridNumY; iy++)
        {
            const vector<int> keypointIndexs = gridKeypoints_[ix][iy];
            if (keypointIndexs.empty())
                continue;
            
            for (int i = 0, N = keypointIndexs.size(); i < N; i++)
            {
                const cv::KeyPoint kpt = unKeypoints_[keypointIndexs[i]];
                if (kpt.octave < min_level || kpt.octave > max_level)
                    continue;
                
                const float distx = kpt.pt.x - u;
                const float disty = kpt.pt.y - v;
                
                if (fabs(distx) < radius && fabs(disty) < radius)
                    indexs.push_back(keypointIndexs[i]);
            }
        }
    }
    
    return indexs;
}

void Frame::computeBow()
{
    // change this according to vocabulary you used 
    if (featVec_.empty() || bowVec_.empty())
        voc_->transform(getDespVector(), bowVec_, featVec_, 3);
}

}





