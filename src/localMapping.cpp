#include "myslam/localMapping.h"
#include "myslam/matcher.h"
#include "myslam/optimizer_ceres.h"
#include "myslam/loopClosing.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <thread>

namespace myslam
{
    
LocalMapping::LocalMapping (Map* map) 
    : map_(map), acceptKeyFrameFlag_(true), stopBAFlag_(false), threadFinishFlag_(false), 
      finishRequestFlag_(false), stopFlag_(false), stopRequestFlag_(false) {}                               

void LocalMapping::run()
{
    threadFinishFlag_ = false;
    
    while (1)
    {       
        setAcceptKeyFrame(false);
        
        if (checkNewKeyFrames())
        {
            processNewKeyFrame();
            cullingMapPoints();
            createNewMapPoints();
            
            if (!checkNewKeyFrames())
                searchInNeighbors();
            
            stopBAFlag_ = false;
            
            if (!checkNewKeyFrames() && !checkStopRequst())
            {
                if (map_->getAllKeyFramesCnt() > 2)
                    Optimizer::solveLocalBAPoseAndPoint(keyframe_curr_, stopBAFlag_, map_);
                
                cullingKeyFrames();
            }
            
            loopCloser_->insertKeyFrame(keyframe_curr_);
        }
        else if (checkStopState())
        {
            while (isStopped() && !checkFinishRequest())
            {
                this_thread::sleep_for(chrono::milliseconds(3));
            }
            
            if (checkFinishRequest())
                break;
        }
        
        setAcceptKeyFrame(true);
        
        if (checkFinishRequest())
            break;
        
        this_thread::sleep_for(chrono::milliseconds(3));
    }
    
    setFinish();

}

void LocalMapping::insertKeyFrame(KeyFrame* keyframe)
{
    unique_lock<mutex> lock(mutexNewKFs_);
    newKeyframes_.push_back(keyframe);
    stopBAFlag_ = true;
}

bool LocalMapping::checkNewKeyFrames()
{
    unique_lock<mutex> lock(mutexNewKFs_);
    return ( !newKeyframes_.empty() );
}

int LocalMapping::inListKeyFrames()
{
    unique_lock<mutex> lock(mutexNewKFs_);
    
    return newKeyframes_.size();
}

void LocalMapping::setAcceptKeyFrame(bool flag)
{
    unique_lock<mutex> lock(mutexAccept_);
    acceptKeyFrameFlag_ = flag;
}

bool LocalMapping::getAcceptStatus()
{
    unique_lock<mutex> lock(mutexAccept_);
    return acceptKeyFrameFlag_;
}

void LocalMapping::processNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mutexNewKFs_);
        keyframe_curr_ = newKeyframes_.front();
        newKeyframes_.pop_front();
    }
    
    keyframe_curr_->computeBow();

    const vector<MapPoint*> mappoints = keyframe_curr_->getMapPoints();
    for (size_t i = 0, N = mappoints.size(); i < N; i++)
    {
        MapPoint* mp = mappoints[i];
        if (mp && !mp->isBad())
        {
            if (!mp->beObserved(keyframe_curr_)) // add by tracking local map
            {
                mp->addObservation(keyframe_curr_, i);
                mp->updateNormalAndDepth();
                mp->computeDescriptor();
            }
            
            else
                addedMapPoints_.push_back(mp);
        }
    }
    
    keyframe_curr_->updateConnections();
    map_->insertKeyFrame(keyframe_curr_);
}

void LocalMapping::createNewMapPoints()
{
    Matcher matcher;
    
    const vector<KeyFrame*> neighbors = keyframe_curr_->getBestCovisibleKFs(10);
    
    Camera* camera = keyframe_curr_->camera_;
    const float fx = camera->fx_;
    const float cx = camera->cx_;
    const float fy = camera->fy_;
    const float cy = camera->cy_;
    const float bf = camera->bf_;
    const float b = camera->b_;
    
    Eigen::Matrix3d K = camera->K_eigen_;
    
    Vector3d Owc = keyframe_curr_->getCamCenter();
    
    SE3 Tcw_curr = keyframe_curr_->getPose();
    Eigen::Matrix3d Rcw1 = Tcw_curr.rotation_matrix();
    Vector3d tcw1 = Tcw_curr.translation();
    
    Mat Tcw1 = (cv::Mat_<float>(3, 4) <<
              Rcw1(0,0), Rcw1(0,1), Rcw1(0,2), tcw1[0],
              Rcw1(1,0), Rcw1(1,1), Rcw1(1,2), tcw1[1],
              Rcw1(2,0), Rcw1(2,1), Rcw1(2,2), tcw1[2]);
    
    int cnt = 0;
    for (int i = 0, N = neighbors.size(); i < N; i++)
    {
        // only search best covisible keyframe if still have newKeyFrames in list
        if (i>0 && checkNewKeyFrames())
            return;
        
        KeyFrame* kf = neighbors[i];
        if (kf->isBad())
            continue;
        
        Vector3d Owi = kf->getCamCenter();
        Vector3d baseline = Owi - Owc;
        float bl = baseline.norm();
        if (bl < b)
            continue;
        
        SE3 Tcwi = kf->getPose();
        Eigen::Matrix3d Rcw2 = Tcwi.rotation_matrix();
        Vector3d tcw2 = Tcwi.translation();
        Mat Tcw2 = (cv::Mat_<float>(3, 4) <<
                    Rcw2(0,0), Rcw2(0,1), Rcw2(0,2), tcw2[0],
                    Rcw2(1,0), Rcw2(1,1), Rcw2(1,2), tcw2[1],
                    Rcw2(2,0), Rcw2(2,1), Rcw2(2,2), tcw2[2]);

        Eigen::Matrix3d F12 = computeF12(Tcw_curr, Tcwi, K);
        
        vector< pair<int, int> > matchIdxs;
        matcher.searchForTriangulation(keyframe_curr_, kf, matchIdxs, F12);
        
        if (matchIdxs.empty())
            continue;
        
        for (int j = 0; j < matchIdxs.size(); j++)
        {   
            const int &idx1 = matchIdxs[j].first;
            const int &idx2 = matchIdxs[j].second;
            
            cv::KeyPoint kp1 = keyframe_curr_->unKeypoints_[idx1];
            cv::KeyPoint kp2 = kf->unKeypoints_[idx2];
            
            const float depth1 = keyframe_curr_->depth_[idx1];
            const float depth2 = kf->depth_[idx2];
            
            const float kp1_ur = keyframe_curr_->uRight_[idx1];
            bool stereo1 = kp1_ur >= 0;
            
            const float kp2_ur = kf->uRight_[idx2];
            bool stereo2 = kp2_ur >= 0;
            
            Vector3d pcam1 = camera->pixel2camera(kp1, 1);
            Vector3d pcam2 = camera->pixel2camera(kp2, 1);
            // check parallax by projection pixel
            
            Vector3d ray1 = Rcw1.transpose() * pcam1;
            Vector3d ray2 = Rcw2.transpose() * pcam2;
            const float cosParallaxRay = ray1.dot(ray2) / (ray1.norm()*ray2.norm());
            
            // check parallax by depth(valid depth with uR >0)
            
            float cosParallaxDepth1 = 2.0f, cosParallaxDepth2 = 2.0f;
            if (stereo1)
                cosParallaxDepth1 = cos( static_cast<float>( 2*atan2(0.5*b, depth1) ) );
            else if(stereo2)
                cosParallaxDepth2 = cos( static_cast<float>( 2*atan2(0.5*b, depth2) ) );
            
            float cosParallaxDepth = min(cosParallaxDepth1, cosParallaxDepth2);
            
            Vector3d p3d;
            
            // if stereo verify ok, but cosRay is less, use cosRay
            // if stereo verify NG, but cosRay is between 0 and 0.9998, use cosRay
            if (cosParallaxRay>0 && cosParallaxRay<cosParallaxDepth
                && (stereo1||stereo2||cosParallaxRay<0.9998))
            {
                Mat A(4,4,CV_32F);
                A.row(0) = static_cast<float>(pcam1[0])*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = static_cast<float>(pcam1[1])*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = static_cast<float>(pcam2[0])*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = static_cast<float>(pcam2[1])*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                Mat x = vt.row(3).t();
                
                // check depth
                if ( fabs(x.at<float>(3)) < 1e-8)
                    continue;
                
                x /= x.at<float>(3);
                
                p3d = Vector3d(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0)); 
            }
            
            // NOTE ORB_SLAM  use original distort keypoint here
            else if (stereo1 && cosParallaxDepth1 < cosParallaxDepth2)
                p3d = camera->pixel2world(kp1, depth1, Tcw_curr);
            else if (stereo2 && cosParallaxDepth2 < cosParallaxDepth1)
                p3d = camera->pixel2world(kp2, depth2, Tcwi);
            else
                continue;
        
            float z1 = Rcw1.row(2).dot(p3d) + tcw1[2];
            if (z1 <= 0)
                continue;
            
            float z2 = Rcw2.row(2).dot(p3d) + tcw2[2];
            if (z2 <= 0)
                continue;


            // check reprojection error
            const float x1 = Rcw1.row(0).dot(p3d) + tcw1[0];
            const float y1 = Rcw1.row(1).dot(p3d) + tcw1[1];
            const float invz1 = 1.0f/z1;
            const float invSigma1 = 1.0f/keyframe_curr_->scaleFactors_[kp1.octave];
            
            const float u1 = fx*x1*invz1 + cx;
            const float v1 = fy*y1*invz1 + cy;
            const float e_u1 = u1 - kp1.pt.x;
            const float e_v1 = v1 - kp1.pt.y;
            const float e1 = e_u1*e_u1 + e_v1*e_v1;
            
            if (!stereo1)
            {
                if (e1*invSigma1*invSigma1 > 5.991f)
                    continue;
            }
            else
            {
                float u1_r = u1 - bf*invz1;
                float e_ur1 = u1_r - kp1_ur;
                float e1r = e1 + e_ur1*e_ur1;
                
                if (e1r*invSigma1*invSigma1 > 7.815f)
                    continue;
            }

            const float x2 = Rcw2.row(0).dot(p3d) + tcw2[0];
            const float y2 = Rcw2.row(1).dot(p3d) + tcw2[1];
            const float invz2 = 1.0f/z2;
            const float invSigma2 = 1.0f/kf->scaleFactors_[kp2.octave];
            
            const float u2 = fx*x2*invz2 + cx;
            const float v2 = fy*y2*invz2 + cy;
            const float e_u2 = u2 - kp2.pt.x;
            const float e_v2 = v2 - kp2.pt.y;
            const float e2 = e_u2*e_u2 + e_v2*e_v2;
            
            if (!stereo2)
            {
                if (e2*invSigma2*invSigma2 > 5.991f)
                    continue;
            }
            else
            {
                const float u2_r = u2 - bf*invz2;
                const float e_ur2 = u2_r - kp2_ur;
                float e2r = e2 + e_ur2*e_ur2;
                
                if (e2r*invSigma2*invSigma2 > 7.815f)
                    continue;
            }

            
            // check scale consistency
            Vector3d n1 = p3d - Owc;
            float dist1 = n1.norm();
            
            Vector3d n2 = p3d - Owi;
            float dist2 = n2.norm();
            
            if ( dist1 < 1e-6 || dist2 < 1e-6)
                continue;
            
            //NOTE: dist2/dist1
            const float distRatio = dist2/dist1;
            const float scaleRatio = keyframe_curr_->scaleFactors_[kp1.octave]/kf->scaleFactors_[kp2.octave];
            const float scaleFactor = 1.5f * keyframe_curr_->scaleFactors_[1];
            
            if (distRatio*scaleFactor < scaleRatio || distRatio > scaleRatio*scaleFactor)
                continue;
            
            // wrap mappoint
            MapPoint* mp(new MapPoint(p3d, keyframe_curr_,idx1, map_));
            mp->addObservation(keyframe_curr_, idx1);
            mp->addObservation(kf, idx2);
            
            keyframe_curr_->addMapPoint(mp, idx1);
            kf->addMapPoint(mp, idx2);
            
            mp->computeDescriptor();
            mp->updateNormalAndDepth();
            map_->insertMapPoint(mp);
            
            addedMapPoints_.push_back(mp);

            cnt++;
        }
    }

}

void LocalMapping::searchInNeighbors()
{
    vector<KeyFrame*> neighbors = keyframe_curr_->getBestCovisibleKFs(10);
    vector<KeyFrame*> expendNeighbors;
    
    for (auto itn = neighbors.begin(), itne= neighbors.end(); itn != itne; itn++)
    {
        KeyFrame* kfn = *itn;
        if (kfn->isBad() || kfn->fuseKFId_ == keyframe_curr_->id_)
            continue;
        
        expendNeighbors.push_back(kfn);
        kfn->fuseKFId_ = keyframe_curr_->id_;
        
        vector<KeyFrame*> secondNeighbors = kfn->getBestCovisibleKFs(5);
        for (auto its = secondNeighbors.begin(), itse = secondNeighbors.end(); its != itse; its++)
        {
            KeyFrame* kfs = *its;
            if (kfs->isBad() || kfs->fuseKFId_ == keyframe_curr_->id_ || kfs->id_ == keyframe_curr_->id_)
                continue;
            
            expendNeighbors.push_back(kfs);
        }
    }
    
    Matcher matcher;
    float threshold = 3.0f;
    vector<MapPoint*> mappoints_curr = keyframe_curr_->getMapPoints();
    for (auto it = expendNeighbors.begin(), ite = expendNeighbors.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        matcher.fuseMapPoints(kf, mappoints_curr, threshold);
    }
    
    vector<MapPoint*> fuseCandidates;
    fuseCandidates.reserve(expendNeighbors.size()*mappoints_curr.size());
    
    for (auto itg = expendNeighbors.begin(); itg != expendNeighbors.end(); itg++)
    {
        KeyFrame* kfg = *itg;
        
        vector<MapPoint*> mappoints_neighbor = kfg->getMapPoints();
        for (auto itk = mappoints_neighbor.begin(), itke = mappoints_neighbor.end(); itk != itke; itk++)
        {
            MapPoint* mp = *itk;
            if (!mp || mp->isBad())
                continue;
            if (mp->fuseForKF_ == keyframe_curr_->id_)
                continue;
            
            mp->fuseForKF_ = keyframe_curr_->id_;
            fuseCandidates.push_back(mp);
        }
    }
    
    matcher.fuseMapPoints(keyframe_curr_, fuseCandidates, threshold);
    
    vector<MapPoint*> mappoints = keyframe_curr_->getMapPoints();
    for (auto it = mappoints.begin(), ite = mappoints.end(); it != ite; it++)
    {
        MapPoint* mp = *it;
        if (mp && !mp->isBad())
        {
            mp->computeDescriptor();
            mp->updateNormalAndDepth();
        }
    }
    
    keyframe_curr_->updateConnections();
}

void LocalMapping::cullingKeyFrames()
{
    int min_obs = 3, cull_cnt = 0;
    float thDepth = keyframe_curr_->camera_->thDepth_;
    
    vector<KeyFrame*> orderedConnectKFs = keyframe_curr_->getOrderedKFs();
    for (auto it = orderedConnectKFs.begin(), ite = orderedConnectKFs.end(); it != ite; it++)
    {
        int mp_cnt = 0, re_obs = 0;
        
        KeyFrame* kf = *it;
        if (kf->isBad() || kf->id_ == 0)
            continue;
        
        const vector<MapPoint*> mappoints = kf->getMapPoints();
        for (size_t i = 0; i < kf->N_; i++)
        {
            MapPoint* mp = mappoints[i];
            if (!mp || mp->isBad())
                continue;
            
            if ( (kf->depth_[i] < 0) || (kf->depth_[i] > thDepth) )
                continue;
            
            mp_cnt++;

            if (mp->getObsCnt() > min_obs)
            {
                const int level = kf->unKeypoints_[i].octave;
                int obskf = 0;
                
                const map<KeyFrame*, size_t> observedKFs = mp->getObservedKFs();
                for (auto itm = observedKFs.begin(), itme = observedKFs.end(); itm != itme; itm++)
                {
                    KeyFrame* kfm = itm->first;
                    if (kfm->isBad() || kfm == kf)
                        continue;
                    
                    const int level_m = kfm->unKeypoints_[itm->second].octave;
                    
                    if (level_m <= level+1)
                    {
                        obskf++;
                        if (obskf >= min_obs)
                            break;
                    }
                }
                
                if (obskf >= min_obs)
                    re_obs++;
            }
        }
        
        if (re_obs > 0.9 * mp_cnt)
        {
            kf->eraseKeyFrame();
            cull_cnt++;
        }
        
    }
}

void LocalMapping::cullingMapPoints()
{    
    list<MapPoint*>::iterator it = addedMapPoints_.begin();
    
    const int minObs = 3;
    const unsigned long currKF_id = keyframe_curr_->id_;
    
    while (it != addedMapPoints_.end())
    {
        MapPoint* mp = *it;
        
        if (mp->isBad())
            it = addedMapPoints_.erase(it);
        else if (mp->getFoundRatio() < 0.25f)
        {
            mp->eraseMapPoint();
            it = addedMapPoints_.erase(it);
        }
        else if(currKF_id > mp->firstAddedIdxofKF_+2 && mp->getObsCnt() <= minObs)
        {
            mp->eraseMapPoint();
            it = addedMapPoints_.erase(it);
        }
        else if (currKF_id > mp->firstAddedIdxofKF_+3)
            it = addedMapPoints_.erase(it);
        else
            it++;
    }
}

Eigen::Matrix3d LocalMapping::computeF12(SE3 &T1w, SE3 &T2w, Eigen::Matrix3d &K)
{
    SE3 T12 = T1w * T2w.inverse();
    Vector3d t12 = T12.translation();
    Eigen::Matrix3d t12x;
    t12x << 0.0, -t12[2], t12[1],
            t12[2], 0.0, -t12[0],
            -t12[1], t12[0], 0.0;
    
    return K.transpose().inverse() * t12x * T12.rotation_matrix() * K.inverse();
}

void LocalMapping::interruptBA()
{
    stopBAFlag_ = true;
}

void LocalMapping::setFinish()
{
    unique_lock<mutex> lock1(mutexFinish_);
    threadFinishFlag_ = true;
    unique_lock<mutex> lock2(mutexStop_);
    stopFlag_ = true;
}

bool LocalMapping::checkFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    return threadFinishFlag_;
}

void LocalMapping::requestFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    finishRequestFlag_ = true;
}

bool LocalMapping::checkFinishRequest()
{
    unique_lock<mutex> lock(mutexFinish_);
    return finishRequestFlag_;
}

void LocalMapping::requestStop()
{
    unique_lock<mutex> lock1(mutexStop_);
    stopRequestFlag_ = true;
    unique_lock<mutex> lock2(mutexNewKFs_);
    stopBAFlag_ = true;
}

bool LocalMapping::checkStopRequst()
{
    unique_lock<mutex> lock(mutexStop_);
    return stopRequestFlag_;
}

bool LocalMapping::checkStopState()
{
    unique_lock<mutex> lock(mutexStop_);
    if (stopRequestFlag_)
    {
        stopFlag_ = true;
        cout << "local mapping stop..." << endl;
        return true;
    }
    
    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mutexStop_);
    return stopFlag_;
}

void LocalMapping::release()
{
    unique_lock<mutex> lock1(mutexStop_);
    unique_lock<mutex> lick2(mutexFinish_);
    
    if (threadFinishFlag_)
        return;
    
    stopFlag_ = false;
    stopRequestFlag_ = false;
    
    //NOTE clear new keyframes
    for (auto it = newKeyframes_.begin(), ite = newKeyframes_.end(); it != ite; it++)
        delete *it;
    newKeyframes_.clear();
    
    cout << "local mapping release.." << endl;
}

}
