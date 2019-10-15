#include "myslam/visualOdometry.h"
#include "myslam/localMapping.h"
#include "myslam/config.h"
#include "myslam/matcher.h"
#include "myslam/optimizer_ceres.h"
#include "myslam/drawer.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <thread>

namespace myslam
{

VisualOdometry::VisualOdometry(Map* map, Camera* camera, Drawer* drawer)
    : state_(INITILIZING), map_(map), keyframe_trackRef_(nullptr), frame_curr_(nullptr),
      frame_last_(nullptr), Tcl_(SE3()), motionModel_(false), lastKeyFrameId_(0), lastRelocateFrameId_(0),
      inliers_num_(0), num_lost_(0), voc_(nullptr), camera_(camera), grayImg_(Mat()), drawer_(drawer)
{
    max_lost_   = Config::get<int>("max_lost");
    int rgbFlag = Config::get<int>("camera_RGB");
    
    rgbFormat_ = rgbFlag;
    maxFrameGap_ = camera->fps_;
    
    num_of_features_  = Config::get<int>("num_of_features");
    scale_factor_     = Config::get<float>("scale_factor");
    level_pyramid_    = Config::get<int>("level_pyramid");
    
    orb_ = new ORB_SLAM2::ORBextractor(num_of_features_, scale_factor_, level_pyramid_, 20, 7);
}

VisualOdometry::~VisualOdometry() {}



void VisualOdometry::run(Mat &rgbImg, Mat &depthImg, string &timeStamp)
{
    timestampDB_.push_back(timeStamp);
    
    createFrame(rgbImg, depthImg, timeStamp);
    
    lastState_ = state_;
    
    unique_lock<mutex> lock(map_->mutexMapUpdate_);
    
    bool track_ok = false;
    
    switch(state_)
    {
        case INITILIZING:
        {
            initVisualOdometry();
            return;
        }
        case LOST:
        {
//             cout << "relocalization..." << endl;
            
            track_ok = relocalization();
            break;
            
        }
        case OK:
        {
            
            track_ok = trackWithMotion();
            
            if (!track_ok)
                track_ok = trackRefKeyFrame();
            
            if (!track_ok)
                track_ok = relocalization();

            break;
        }
    }

    frame_curr_->keyframe_trackRef_ = keyframe_trackRef_;
    
    if (track_ok)
        track_ok = trackLocalMap();
    
    drawer_->updateCurrFrame(this);

    if (track_ok)
    {
        num_lost_ = 0;
        state_ = OK;
        
        drawer_->setCurrPose(frame_curr_->Tcw_);
        
        if (frame_last_->poseExist_)
        {
            Tcl_ = frame_curr_->Tcw_ * frame_last_->Tcw_.inverse();
            motionModel_ = true;
        }
        else
        {
            Tcl_ = SE3();
            motionModel_ = false;
        }
        
        cullingTempMapPoints();  
        
        if (needNewKeyFrame())
            createNewKeyFrame();
        
        cullingOutliersOfFrame();
        
    }
    else
    {
        num_lost_++;
        state_ = LOST;
        Tcl_ = SE3();
        motionModel_ = false;
        
        map_->lostFrames_.push_back(frame_curr_);
//         cout << "lost time: " << num_lost_ << endl;
    }
    
    if (!frame_curr_->keyframe_trackRef_)
    frame_curr_->keyframe_trackRef_ = keyframe_trackRef_;
    
    stateDB_.push_back(state_ == OK);
    frame_last_ = frame_curr_;
    
    if (frame_curr_->poseExist_)
    {
        SE3 Tcr = frame_curr_->Tcw_ * keyframe_trackRef_->Tcw_.inverse();
        TcrDB_.push_back(Tcr);
        trackRefKeyFrameDB_.push_back(keyframe_trackRef_);
    }
    else
    {
        TcrDB_.push_back(TcrDB_.back());
        trackRefKeyFrameDB_.push_back(trackRefKeyFrameDB_.back());
    }
    
}

void VisualOdometry::createFrame(Mat &rgbImg, Mat &depthImg, string timeStamp)
{
    grayImg_ = rgbImg;
    if (rgbImg.channels() == 3)
    {
        if (rgbFormat_)
            cvtColor(grayImg_, grayImg_, CV_RGB2GRAY);
        else
            cvtColor(grayImg_, grayImg_, CV_BGR2GRAY);
    }
    else if (rgbImg.channels() == 4)
    {
        if (rgbFormat_)
            cvtColor(grayImg_, grayImg_, CV_RGBA2GRAY);
        else
            cvtColor(grayImg_, grayImg_, CV_BGRA2GRAY);
    }
    
    float invDepthScale = 1.0f / camera_->depth_scale_;
    depthImg.convertTo(depthImg, CV_32F, invDepthScale);
    
    frame_curr_ = new Frame(grayImg_, depthImg, timeStamp, camera_, orb_);
    
    frame_curr_->voc_ = voc_;
}

void VisualOdometry::initVisualOdometry()
{
    if (state_ == INITILIZING)
    {
        frame_curr_->setPose(SE3());
        
        KeyFrame* kf = new KeyFrame(frame_curr_, map_);
        
        // create mappoints for initial frame
        int cnt = 0;
        for (int i = 0; i < frame_curr_->N_; i++)
        {
            float d = frame_curr_->depth_[i];
            if (d < 0)
                continue;
            
            Vector3d p_world = camera_->pixel2world(frame_curr_->unKeypoints_[i], d, frame_curr_->Tcw_);
            MapPoint* point = new MapPoint(p_world, kf, i, map_);
            
            point->addObservation(kf, i);
            point->computeDescriptor();
            point->updateNormalAndDepth();
            
            kf->addMapPoint(point, i);
            frame_curr_->mappoints_[i] = point;
            map_->insertMapPoint(point);
            
            cnt++;
        }
        
        cout << "new map created with " << cnt << " mappoints" << endl;
        
        localMapper_->insertKeyFrame(kf);
        
        localKeyframes_.push_back(kf);
        localMappoints_ = map_->getAllMapPoints();
        
        keyframe_trackRef_ = kf;
        frame_curr_->keyframe_trackRef_ = kf;
        
        frame_last_ = frame_curr_;
        lastKeyFrameId_ = frame_curr_->id_;
        
        trackRefKeyFrameDB_.push_back(keyframe_trackRef_);
        TcrDB_.push_back(SE3());
        stateDB_.push_back(true);
        
        drawer_->updateCurrFrame(this);
        drawer_->setCurrPose(frame_curr_->Tcw_);

        state_ = OK;
        
    }
}

bool VisualOdometry::trackWithMotion()
{
    if (!motionModel_)
        return false;
    
    if (frame_curr_->id_ < lastRelocateFrameId_ + 2)
        return false;
    
    recoverLastFrame();
    updateLastFrame();
 
    frame_curr_->setPose(Tcl_ * frame_last_->Tcw_);
    
    Matcher matcher;
    float radius = 15.0f;
    int match_num = matcher.searchByProjection(frame_curr_, frame_last_, radius);
    
    if (match_num < 20)
    {
        fill(frame_curr_->mappoints_.begin(), frame_curr_->mappoints_.end(), static_cast<MapPoint*>(nullptr));
        match_num = matcher.searchByProjection(frame_curr_, frame_last_, 2*radius);
    }
    
    if (match_num < 20)    
        return false;
    
    Optimizer::solvePoseOnlySE3 (frame_curr_);
    int inliers_num = cullingOutliersBeforeLocalMap();
    
    return inliers_num >= 10;
}

bool VisualOdometry::trackRefKeyFrame()
{    
    int match_num;

    frame_curr_->computeBow();
    
    Matcher matcher(0.7);
    vector<MapPoint*> mappointsMatches;
    match_num = matcher.searchByBoW(keyframe_trackRef_, frame_curr_, mappointsMatches);
    
    if (match_num < 15)
        return false;

    frame_curr_->mappoints_ = mappointsMatches;
    frame_curr_->setPose(frame_last_->Tcw_);
    
    Optimizer::solvePoseOnlySE3 (frame_curr_);
    int inliers_num = cullingOutliersBeforeLocalMap();
    
    return inliers_num >= 10;
}

bool VisualOdometry::trackLocalMap()
{
    map_->setLocalMapPoints(localMappoints_);

    updateLocalKeyFrames();  
    updateLocalMapPoints();
    searchLocalMapPoints();
    
    Optimizer:: solvePoseOnlySE3(frame_curr_);
    
    inliers_num_ = 0;
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_curr_->mappoints_[i];
        
        //NOTE DO NOT set all outliers null, some of them may be tracked from map
        if (mp && !frame_curr_->outliers_[i])
        {
            mp->addFound();
            
            // inliers tracked from map
            if (mp->getObsCnt()>0)
                inliers_num_++;
        }
    }
    
//     cout << "observed inliers num: " << inliers_num_ << endl;

    if (frame_curr_->id_ < lastRelocateFrameId_+maxFrameGap_  && inliers_num_ < 50)
        return false;
    
    return inliers_num_ >= 30;
}

bool VisualOdometry::relocalization()
{
    frame_curr_->computeBow();
    vector<KeyFrame*> candidates = map_->detectRelocalizationCandidates(frame_curr_);
    if (candidates.empty())
        return false;
    
    for (auto it = candidates.begin(); it != candidates.end(); it++)
    {
        KeyFrame* kf = *it;
        if (kf->isBad())
            continue;
        
        vector<MapPoint*> mappointsMatches;
        Matcher matcherBow(0.75);
        int searchMatch_cnt = 0;
        searchMatch_cnt = matcherBow.searchByBoW(kf, frame_curr_, mappointsMatches);
        if (searchMatch_cnt < 15)
            continue;
        
        set<MapPoint*> found;
        int pnpInlier_cnt = 0;
        pnpInlier_cnt = poseEstimateByPnP(frame_curr_, found, mappointsMatches);
        if (pnpInlier_cnt < 10)
            continue;
        
        inliers_num_ = 0;
        inliers_num_ = Optimizer::solvePoseOnlySE3(frame_curr_);
        
        if (inliers_num_ < 10)
            continue;
        
        for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
        {
            if (frame_curr_->outliers_[i])
                frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
        }
        
        
        if (inliers_num_ < 50)
        {
            Matcher matcherProj;
            int firstAddition = matcherProj.searchByProjection(frame_curr_, kf, 10, 100, found);
            
            if (inliers_num_ + firstAddition >= 50)
            {
                inliers_num_ = Optimizer::solvePoseOnlySE3(frame_curr_);
                
                if (inliers_num_ > 30 && inliers_num_ < 50)
                {
                    found.clear();
                    
                    for (int im = 0, N = frame_curr_->mappoints_.size(); im < N; im++)
                    {
                        if (frame_curr_->mappoints_[im])
                                found.insert(frame_curr_->mappoints_[im]);
                    }
                    
                    int secondAddition = matcherProj.searchByProjection(frame_curr_, kf, 3, 60, found);
                    
                    if (inliers_num_ + secondAddition >= 50)
                    {
                        inliers_num_ = Optimizer::solvePoseOnlySE3(frame_curr_);
                        
                        for (int j = 0, N = frame_curr_->mappoints_.size(); j < N; j++)
                        {
                            if (frame_curr_->outliers_[j])
                                frame_curr_->mappoints_[j] = static_cast<MapPoint*>(nullptr);
                        }
                    }
                }
            }
        }
        
        if (inliers_num_ >= 50)
        {
            lastRelocateFrameId_ = frame_curr_->id_;
            return true;
        }
    }
    
    return false;
}

bool VisualOdometry::needNewKeyFrame()
{
    if (localMapper_->isStopped() || localMapper_->checkStopRequst())
        return false;
    
    const int keyframe_cnt = map_->getAllKeyFramesCnt();
    if (frame_curr_->id_ < lastRelocateFrameId_+ maxFrameGap_ && keyframe_cnt > maxFrameGap_)
        return false;

    int minObs = 3;
    if (keyframe_cnt <= 2)
        minObs = 2;
    int refMatches = keyframe_trackRef_->trackedMapPoints(minObs); 
    const float refRatio = static_cast<float>(inliers_num_) / static_cast<float>(refMatches);
    bool refWeak = refRatio < 0.25f || refMatches < 100;
    
    int map_cnt = 0, total_cnt = 0;
    const float thDepth = camera_->thDepth_;
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        const float d = frame_curr_->depth_[i];
        if (d > 0 && d < thDepth)
        {
            total_cnt++;
            MapPoint* mp = frame_curr_->mappoints_[i];
            
            // obsCnt > 0 aka mappoints can be observed by other keyframes
            // map_cnt is less, means most mappoints added by tracking motion, risky
            if (mp && mp->getObsCnt()> 0)
                map_cnt++;
        }
    }
    
    float map_threshold = 0.35f;
    if (inliers_num_ > 300)
        map_threshold = 0.20f;
    float mapRatio = float(map_cnt) / float(total_cnt + 1e-5);
    
    bool trackWeak = mapRatio < 0.3f;

    float refThreshold = 0.75f;
    if (keyframe_cnt < 2)
        refThreshold = 0.40f;
    
    bool trackGap = (frame_curr_->id_ >= lastKeyFrameId_+maxFrameGap_) || localMapper_->getAcceptStatus();
    bool trackVerify = refRatio < refThreshold || mapRatio < map_threshold;
    bool trackReserve = trackGap && trackVerify;
    
    if (trackWeak || refWeak || trackReserve)
    {
        if (localMapper_->getAcceptStatus())
            return true;
        else
        {
            localMapper_->interruptBA();
            
            if (localMapper_->inListKeyFrames() < 3)
                return true;
            else
                return false;
        }
    }
    else
        return false;
}

void VisualOdometry::createNewKeyFrame()
{
    KeyFrame* kf = new KeyFrame(frame_curr_, map_);
    keyframe_trackRef_ = kf;
    frame_curr_->keyframe_trackRef_ = kf;
    
    vector< pair<float,int> > depthPairs;
    depthPairs.reserve(frame_curr_->N_);
    
    for (int i = 0; i < frame_curr_->N_; i++)
    {
        float d = frame_curr_->depth_[i];
        if (d > 0)
            depthPairs.push_back(make_pair(d,i));
    }
    
    if (!depthPairs.empty())
    {
        sort(depthPairs.begin(), depthPairs.end());

        float threshold = camera_->thDepth_;
        int point_cnt = 0;
        for (int j = 0; j < depthPairs.size(); j++)
        {
            int idx = depthPairs[j].second;
            float d = depthPairs[j].first;
            
            MapPoint* mp = frame_curr_->mappoints_[idx];
            
            // obsCnt < 1 , aka added by tracking last frame(which update with 0 obs mappoints) 
            if (!mp || (mp && mp->getObsCnt() < 1))
            {
                Vector3d p_world = camera_->pixel2world(frame_curr_->unKeypoints_[idx], d, frame_curr_->Tcw_);
                MapPoint* point = new MapPoint(p_world, kf, idx, map_);
                
                kf->replaceMapPoint(point, idx);
                frame_curr_->mappoints_[idx] = point;
                
                point->addObservation(kf, idx);
                point->computeDescriptor();
                point->updateNormalAndDepth();
                
                map_->insertMapPoint(point);
                
                point_cnt++;
            }
            
            if (d > threshold && point_cnt > 100)
                break;
        }        
    }

    lastKeyFrameId_ = frame_curr_->id_;
    localMapper_->insertKeyFrame(kf);
}

void VisualOdometry::recoverLastFrame()
{
    int cnt = 0;
    for (int i = 0, N = frame_last_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_last_->mappoints_[i];
        if (!mp)
            continue;
        
        MapPoint* rep = mp->getReplacedMapPoint();
        if (rep)
        {
            frame_last_->mappoints_[i] = rep;
            cnt++;
        }
    }
    
}

void VisualOdometry::printPose(SE3 &Tcw)
{
    Eigen::Quaterniond r(Tcw.rotation_matrix());
    cout << Tcw.translation().transpose() << " " << r.coeffs().transpose() << endl;
}

void VisualOdometry::updateLastFrame()
{
    // update last frame pose due to localBA
    KeyFrame* trackRef_last = frame_last_->keyframe_trackRef_;
    SE3 Tlr = TcrDB_.back();
    frame_last_->setPose(Tlr * trackRef_last->getPose());
    
    // already add mappoints when create keyframe
    if (frame_last_->id_ == lastKeyFrameId_)
        return;
    
    // read depth value and sort
    vector< pair<float,int> > depthPairs;
    depthPairs.reserve(frame_last_->N_);
    for (int i = 0; i < frame_last_->N_; i++)
    {
        float d = frame_last_->depth_[i];
        
        if (d > 0)
            depthPairs.push_back(make_pair(d,i));
    }
    
    if (depthPairs.empty())
        return;
    
    sort(depthPairs.begin(), depthPairs.end());
    
    float threshold = camera_->thDepth_;
    int point_cnt = 0;
    for (int i = 0; i < depthPairs.size(); i++)
    {
        int idx = depthPairs[i].second;
        float d = depthPairs[i].first;
        
        MapPoint* mp = frame_last_->mappoints_[idx];
        if (!mp || (mp && mp->getObsCnt() < 1))
        {
            Vector3d p_world = camera_->pixel2world(frame_last_->unKeypoints_[idx], d, frame_last_->Tcw_);
            MapPoint* point = new MapPoint(p_world, frame_last_, idx, map_);
            
            frame_last_->mappoints_[idx] = point;
            tempMappoints_.push_back(point);
            
            point_cnt++;
        }
        
        if (d > threshold && point_cnt > 100)
            break;
    }
}

void VisualOdometry::updateLocalKeyFrames()
{    
    map<KeyFrame*, int> keyframe_counter;
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_curr_->mappoints_[i];
        if (mp)
        {
            if (!mp->isBad())
            {
                const map<KeyFrame*, size_t> observations = mp->getObservedKFs();
                for (auto it = observations.begin(), ite = observations.end(); it != ite; it++)
                {
                    keyframe_counter[it->first]++;
                }
            }
            else
                frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
        }
    }
    
    if (keyframe_counter.empty())
        return;
    
    int observed_max = 0;
    KeyFrame* keyframe_best = static_cast<KeyFrame*>(nullptr);
    
    localKeyframes_.clear();
    localKeyframes_.reserve(3*keyframe_counter.size());
    
    for (auto it = keyframe_counter.begin(), ite = keyframe_counter.end(); it != ite; it++)
    {
        KeyFrame* kf = it->first;
        if (kf->isBad())
            continue;
        
        if (it->second > observed_max)
        {
            observed_max = it->second;
            keyframe_best = kf;
        }
        
        localKeyframes_.push_back(it->first);
        kf->trackFrameId_ = frame_curr_->id_;
    }
    
    for (auto itls = localKeyframes_.begin(), itle = localKeyframes_.end(); itls != itle; itls++)
    {
        if (localKeyframes_.size() > 80)
            break;
        
        KeyFrame* kf = *itls;
        
        // search covisible graph
        const vector<KeyFrame*> neighbors = kf->getBestCovisibleKFs(10);
        for (auto itn = neighbors.begin(), itne = neighbors.end(); itn != itne; itn++)
        {
            KeyFrame* kfn = *itn;
            if (kfn->isBad())
                continue;
            
            if (kfn->trackFrameId_ != frame_curr_->id_ )
            {
                localKeyframes_.push_back(kfn);
                kfn->trackFrameId_ = frame_curr_->id_;
                break;
            }
        }            
        
        // search spanning tree, children
        const set<KeyFrame*> children = kf->getChildren();
        for (auto itc = children.begin(), itce = children.end(); itc != itce; itc++)
        {
            KeyFrame* kfc = *itc;
            if (kfc->isBad())
                continue;
            
            if (kfc->trackFrameId_ != frame_curr_->id_)
            {
                localKeyframes_.push_back(kfc);
                kfc->trackFrameId_ = frame_curr_->id_;
                break;
            }
        }            
        
        // search spanning tree, parent
        KeyFrame* parent = kf->getParent();
        if (parent && parent->trackFrameId_ != frame_curr_->id_)
        {
            if (!parent->isBad())
            {
                localKeyframes_.push_back(parent);
                parent->trackFrameId_ = frame_curr_->id_;
            }
        }
    }
    
    if (keyframe_best)
    {
        keyframe_trackRef_ = keyframe_best;
        frame_curr_->keyframe_trackRef_ = keyframe_best;
    }

}

void VisualOdometry::updateLocalMapPoints()
{
    localMappoints_.clear();
    
    for (auto it = localKeyframes_.begin(), ite = localKeyframes_.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        if (kf->isBad())
            continue;
        
        const vector<MapPoint*> mappoints = kf->getMapPoints();
        for (int i = 0, N = mappoints.size(); i < N; i++)
        {
            MapPoint* mp = mappoints[i];
            if (!mp || mp->isBad())
                continue;
            
            if ( mp->trackIdxOfFrame_ != frame_curr_->id_ )
            {
                localMappoints_.push_back(mp);
                mp->trackIdxOfFrame_ = frame_curr_->id_;                
            }
        }
    }
}

void VisualOdometry::searchLocalMapPoints()
{
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_curr_->mappoints_[i];
        if (mp)
        {
            // if bad, set null and update when create new mappoints
            if (mp->isBad())
                frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
            else
            {
                mp->addVisible();
                mp->visualIdxOfFrame_ = frame_curr_->id_;
                mp->trackInLocalMap_ = false;
            }
        }
    }
    
    
    int match_cnt = 0;
    for (auto it = localMappoints_.begin(); it != localMappoints_.end(); it++)
    {
        MapPoint* mp = *it;
        if (mp->isBad())
            continue;
        
        if (mp->visualIdxOfFrame_ == frame_curr_->id_)
            continue;
        
        if (frame_curr_->isInFrame(mp))
        {
            mp->addVisible();
            match_cnt++;
        }
    }
    
    if (match_cnt > 0)
    {
        Matcher matcher(0.8);
        float thRadius = 3.0f;
        
        if (frame_curr_->id_ < lastRelocateFrameId_+2)
            thRadius = 5.0f;
        
        match_cnt = matcher.searchByProjection(frame_curr_, localMappoints_, thRadius);
    }
    
}

int VisualOdometry::poseEstimateByPnP(Frame* frame, set<MapPoint*> &found, vector<MapPoint*> &mappointsMatches)
{    
    Mat K = camera_->K_;
    
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    
    vector<int> source_idxs;
    for (int i = 0, N =  mappointsMatches.size(); i < N; i++)
    {
        MapPoint* mp = mappointsMatches[i];
        if (mp)
        {
            if (mp->isBad())
                mappointsMatches[i] = static_cast<MapPoint*>(nullptr);
            else
            {
                Vector3d pos = mp->getPose();
                pts_3d.push_back( cv::Point3f(pos[0], pos[1], pos[2]) );
                pts_2d.push_back( frame->unKeypoints_[i].pt );
                source_idxs.push_back(i);
            }
        }
    }
    
    if (source_idxs.empty())
        return 0;
    
    Mat R, r, t;
    vector<int> inliers;
    cv::solvePnPRansac( pts_3d, pts_2d, K, Mat(), r, t, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
    
    if (inliers.empty())
        return 0;
    
    // culling outliers
    for (int i = 0; i < inliers.size(); i++)
    {
        int idx = source_idxs[inliers[i]];

        frame->mappoints_[idx] = mappointsMatches[idx];
        found.insert(frame->mappoints_[idx]);
    }
    
    cv::Rodrigues(r, R);
    SE3 Tcw = poseRtToSE3(R, t);
    
    frame->setPose(Tcw);
    
    return inliers.size();
}

SE3 VisualOdometry::poseRtToSE3(const Mat& R_, const Mat& t_)
{
    Eigen::Matrix3d R;
    R << R_.at<double>(0,0), R_.at<double>(0,1), R_.at<double>(0,2),
         R_.at<double>(1,0), R_.at<double>(1,1), R_.at<double>(1,2),
         R_.at<double>(2,0), R_.at<double>(2,1), R_.at<double>(2,2);
    Vector3d t( t_.at<double>(0,0), t_.at<double>(1,0), t_.at<double>(2,0) );
    
    return SE3(R, t);
}

void VisualOdometry::cullingTempMapPoints()
{
    if (tempMappoints_.empty())
        return;
    
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_curr_->mappoints_[i];

        if (mp && mp->getObsCnt() < 1)
        {
            frame_curr_->outliers_[i] = false;
            frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
        }
    }
    
    for (auto it = tempMappoints_.begin(), ite = tempMappoints_.end(); it != ite; it++)
    {
        MapPoint* mp = *it;
        delete mp;
    }
    
    tempMappoints_.clear();
}

int VisualOdometry::cullingOutliersBeforeLocalMap()
{
    int observedInliers_num = 0;
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_curr_->mappoints_[i];
        if (!mp)
            continue;
        
        if (frame_curr_->outliers_[i])
        {
            frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
            frame_curr_->outliers_[i] = false; // since mappoint null, reset false
            
            mp->trackInLocalMap_ = false;
            mp->visualIdxOfFrame_ = frame_curr_->id_; 
        }
        else if (mp->getObsCnt()>0)
            observedInliers_num++;
    }
    
    return observedInliers_num;
}

void VisualOdometry::cullingOutliersOfFrame()
{
    for (int i = 0, N = frame_curr_->mappoints_.size(); i < N; i++)
    {
        if (frame_curr_->mappoints_[i] && frame_curr_->outliers_[i])
            frame_curr_->mappoints_[i] = static_cast<MapPoint*>(nullptr);
    }
}

}





