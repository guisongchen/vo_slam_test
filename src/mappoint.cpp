#include "myslam/mappoint.h"
#include "myslam/keyframe.h"
#include "myslam/matcher.h"
#include "myslam/map.h"

namespace myslam
{
unsigned long factory_id_ = 0;
mutex MapPoint::mutexOptimizer_;

MapPoint::MapPoint(const Vector3d& postion, Frame* frame, size_t idx, Map* map) :
    pos_(postion), keyFrame_ref_(nullptr), replacedMapPoint_(nullptr), map_(map), idx_(idx),
    trackInLocalMap_(false), found_cnt_ (1), visible_cnt_ (1), observe_cnt_(0), 
    trackIdxOfFrame_(frame->id_), visualIdxOfFrame_(frame->id_), loopPointIdxOfKF_(0),
    loopCorrectByKF_(0), correctReference_(0), localBAKFId_(0), minDistance_(0.0), maxDistance_(0.0),
    trackScaleLevel_(-1), firstAddedIdxofKF_(-1), fuseForKF_(0), badFlag_(false)
{
    Vector3d Ow = frame->Ow_;
    normalVector_ = pos_ - Ow;
    const float dist = normalVector_.norm();
    normalVector_.normalize();
    
    const int level = frame->unKeypoints_[idx].octave;
    const float scaleFactor = frame->scaleFactors_[level];
    const int totalLevels = frame->scaleFactors_.size();
    
    maxDistance_ = dist*scaleFactor;
    minDistance_ = maxDistance_/frame->scaleFactors_[totalLevels-1];
    
    descriptor_ = frame->descriptors_.row(idx_).clone();
    
    unique_lock<mutex> lock(map->mutexCreateMapPoint_);
    id_ = factory_id_++;
}

MapPoint::MapPoint(const Vector3d& postion, KeyFrame* keyframe, size_t idx, Map* map) :
    pos_(postion), keyFrame_ref_(keyframe), replacedMapPoint_(nullptr), map_(map), idx_(idx), 
    trackInLocalMap_(false), found_cnt_ (1), visible_cnt_ (1), observe_cnt_(0),
    trackIdxOfFrame_(keyframe->orgFrameId_), visualIdxOfFrame_(keyframe->orgFrameId_), loopPointIdxOfKF_(0), 
    loopCorrectByKF_(0), correctReference_(0), localBAKFId_(0), minDistance_(0.0), maxDistance_(0.0),
    trackScaleLevel_(-1), firstAddedIdxofKF_(keyframe->id_), fuseForKF_(0), badFlag_(false)
{
    normalVector_ = pos_ - keyframe->getCamCenter();
    normalVector_.normalize();
    
    descriptor_ = keyframe->descriptors_.row(idx_).clone();
    
    unique_lock<mutex> lock(map->mutexCreateMapPoint_);
    id_ = factory_id_++;
}

void MapPoint::addObservation(KeyFrame* keyframe, size_t idx)
{
    unique_lock<mutex> lock(mutexFeature_);
    if (observedKFs_.count(keyframe))
        return;
    
    observedKFs_[keyframe] = idx;
    
    if (keyframe->uRight_[idx] >= 0)
        observe_cnt_ += 2;
    else
        observe_cnt_++;
}

void MapPoint::updateNormalAndDepth()
{
    map<KeyFrame*, size_t> observedKFs;
    KeyFrame* keyFrame_ref;
    Vector3d pos;
    
    {
        unique_lock<mutex> lock1(mutexFeature_);
        unique_lock<mutex> lock2(mutexPose_);
        
        if (badFlag_)
            return;
        
        observedKFs = observedKFs_;
        keyFrame_ref = keyFrame_ref_;
        pos = pos_;
    }
    
    // BA may erase keyframe
    if (!keyFrame_ref)
        return;

    if (observedKFs.empty())
        return;
    Vector3d normal(Vector3d::Zero());
    
    int n = 0;
    for (auto it = observedKFs.begin(); it != observedKFs.end(); it++)
    {
        KeyFrame* kf = it->first;
        
        Vector3d ni = pos - kf->getCamCenter();
        normal = normal + ni/ni.norm();
        n++;
    }
    
    // direction of reference keyframe
    Vector3d line = pos - keyFrame_ref->getCamCenter();
    float dist = line.norm();
    int level = keyFrame_ref->unKeypoints_[observedKFs[keyFrame_ref]].octave;
    float levelScaledFactor = keyFrame_ref->scaleFactors_[level];
    int levelNum = keyFrame_ref->scaleFactors_.size();
    
    {
        unique_lock<mutex> lock3(mutexPose_);
        
        maxDistance_ = dist * levelScaledFactor;
        minDistance_ = maxDistance_ / keyFrame_ref->scaleFactors_[levelNum - 1];
        normalVector_ = normal / n;
    }
}

void MapPoint::computeDescriptor()
{
    vector<Mat> desp;
    map<KeyFrame*, size_t> observedKFs;
    
    {
        unique_lock<mutex> lock(mutexFeature_);
        if (badFlag_)
            return;
        observedKFs = observedKFs_;
    }
    
    
    if (observedKFs.empty())
        return;
    
    desp.reserve(observedKFs.size());
    for (auto it = observedKFs.begin(); it != observedKFs.end(); it++)
    {
        KeyFrame* kf = it->first;
        if (!kf->isBad())
            desp.push_back(kf->descriptors_.row(it->second));
    }
    
    if (desp.empty())
        return;
    
    const size_t N = desp.size();
    vector< vector<float> > distances;
    distances.resize(N, vector<float>(N, 0));
    
    for (size_t i = 0; i < N; i++)
    {
        distances[i][i] = 0;
        for (size_t j = i+1; j < N; j++)
        {
            int distij = Matcher::computeDistance(desp[i], desp[j]);
            distances[i][j] = distij;
            distances[j][i] = distij;
        }
    }
    
    int bestMid = 256;
    int bestIdx = 0;
    for (size_t i = 0; i < N; i++)
    {
        vector<int> dist(distances[i].begin(), distances[i].end());
        sort(dist.begin(), dist.end());
        
        int mid = dist[ int(0.5 * (N-1)) ];
        if (mid < bestMid)
        {
            bestMid = mid;
            bestIdx = i;
        }
    }
    
    {
        unique_lock<mutex> lock(mutexFeature_);
        descriptor_ = desp[bestIdx].clone();
    }
}


int MapPoint::predictScale(const float& currDist, Frame* frame)
{
    float ratio;
    {
        unique_lock<mutex> lock(mutexPose_);
        ratio = maxDistance_ / currDist;
    }
    int scale = ceil( log(ratio)/log(frame->scaleFactors_[1]) );
    if (scale < 0)
        scale = 0;
    else if (scale >= frame->scaleFactors_.size())
        scale = frame->scaleFactors_.size() - 1;
    
    return scale;
}

int MapPoint::predictScale(const float& currDist, KeyFrame* kf)
{
    float ratio;
    {
        unique_lock<mutex> lock(mutexPose_);
        ratio = maxDistance_ / currDist;
    }
    int scale = ceil( log(ratio)/log(kf->scaleFactors_[1]) );
    if (scale < 0)
        scale = 0;
    else if (scale >= kf->scaleFactors_.size())
        scale = kf->scaleFactors_.size() - 1;
    
    return scale;
}

void MapPoint::replaceMapPoint(MapPoint* mp)
{
    if (mp->id_ == this->id_)
        return;
    
    int visible, found;
    map<KeyFrame*, size_t> obs;
    
    {
        unique_lock<mutex> lock1(mutexFeature_);
        unique_lock<mutex> lock2(mutexPose_);
        
        obs = observedKFs_;
        observedKFs_.clear();
        badFlag_ = true;
        
        visible = visible_cnt_;
        found = found_cnt_;
        replacedMapPoint_ = mp;
    }
    
    for (auto it = obs.begin(); it != obs.end(); it++)
    {
        KeyFrame* kf = it->first;
        
        if (!mp->beObserved(kf))
        {
            kf->replaceMapPoint(mp, it->second);
            mp->addObservation(kf, it->second);
        }
        else
            kf->setMapPointNull(it->second);
    }
    
    mp->addFound(found);
    mp->addVisible(visible);
    mp->computeDescriptor();
    
    map_->eraseMapPoint(this);
}

MapPoint* MapPoint::getReplacedMapPoint()
{
    unique_lock<mutex> lock1(mutexFeature_);
    unique_lock<mutex> lock2(mutexPose_);
    
    return replacedMapPoint_;
}

bool MapPoint::beObserved(myslam::KeyFrame* kf)
{
    unique_lock<mutex> lock(mutexFeature_);
    return observedKFs_.count(kf);
}

void MapPoint::addFound(const int cnt)
{
    unique_lock<mutex> lock(mutexFeature_);
    found_cnt_ += cnt;
}

void MapPoint::addVisible(const int cnt)
{
    unique_lock<mutex> lock(mutexFeature_);
    visible_cnt_ += cnt;
}

int MapPoint::getIndexInKeyFrame(KeyFrame* keyframe)
{
    unique_lock<mutex> lock(mutexFeature_);
    if (observedKFs_.count(keyframe))
        return observedKFs_[keyframe];
    else
        return -1;
}

map<KeyFrame*, size_t> MapPoint::getObservedKFs()
{
    unique_lock<mutex> lock(mutexFeature_);
    return observedKFs_;
}

Eigen::Vector3d MapPoint::getPose()
{
    unique_lock<mutex> lock(mutexPose_);
    return pos_;
}

void MapPoint::setPose(const Vector3d &pos)
{
    unique_lock<mutex> lock1(mutexOptimizer_);
    unique_lock<mutex> lock2(mutexPose_);
    pos_ = pos;
}

Mat MapPoint::getDescriptor()
{
    unique_lock<mutex> lock(mutexFeature_);
    return descriptor_;
}

Vector3d MapPoint::getNormalVector()
{
    unique_lock<mutex> lock(mutexFeature_);
    return normalVector_;
}

int MapPoint::getObsCnt()
{
    unique_lock<mutex> lock(mutexFeature_);
    return observe_cnt_;
}

float MapPoint::getFoundRatio()
{
    unique_lock<mutex> lock(mutexFeature_);
    return float(found_cnt_)/float(visible_cnt_);
}

void MapPoint::eraseObservedKF(KeyFrame* kf)
{
    bool eraseFlag = false;
    {
        unique_lock<mutex> lock(mutexFeature_);
        
        if (observedKFs_.count(kf))
        {
            int idx = observedKFs_[kf];
            if (kf->uRight_[idx] >= 0)
                observe_cnt_ -= 2;
            else
                observe_cnt_--;
            
            observedKFs_.erase(kf);
            
            // make sure mappoint be observed, choosen form observedKFs 
            if (keyFrame_ref_ == kf)
                keyFrame_ref_ = observedKFs_.begin()->first;
            
            if(observe_cnt_ <= 2)
                eraseFlag = true;
        }
    }
    
    if (eraseFlag)
        eraseMapPoint();
}

void MapPoint::eraseMapPoint()
{
    map<KeyFrame*, size_t> observedKFs;
    {
        unique_lock<mutex> clock1(mutexFeature_);
        unique_lock<mutex> clock2(mutexPose_);
        
        badFlag_ = true;
        observedKFs = observedKFs_;
        observedKFs_.clear();
    }
    
    for (auto it = observedKFs.begin(), ite = observedKFs.end(); it != ite; it++)
    {
        KeyFrame* kf = it->first;
        kf->setMapPointNull(it->second);
    }
    
    map_->eraseMapPoint(this); // only delete pointer, need to release memory
}

bool MapPoint::isBad()
{
    unique_lock<mutex> clock1(mutexFeature_);
    unique_lock<mutex> clock2(mutexPose_);
        
    return badFlag_;
}

float MapPoint::getMinDistanceThreshold()
{
    unique_lock<mutex> lock(mutexPose_);
    return 0.8f*minDistance_;
}

float MapPoint::getMaxDistanceThreshold()
{
    unique_lock<mutex> lock(mutexPose_);
    return 1.2f*maxDistance_;
}


                       
}
