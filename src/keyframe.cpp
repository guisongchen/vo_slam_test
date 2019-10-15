#include "myslam/keyframe.h"
#include "myslam/map.h"

namespace myslam
{
    
unsigned long keyframe_id = 0;
    
KeyFrame::KeyFrame(Frame* frame, Map* map)
    : orgFrameId_(frame->id_), trackFrameId_(frame->id_), timeStamp_(frame->timeStamp_),
      camera_(frame->camera_), Tcw_(frame->Tcw_), Tcp_(SE3()), Ow_(frame->Ow_),firstConnect_(true),
      parent_(nullptr), keypoints_(frame->keypoints_), unKeypoints_(frame->unKeypoints_),
      depth_(frame->depth_), uRight_(frame->uRight_), descriptors_(frame->descriptors_),
      mappoints_(frame->mappoints_), scaleFactors_(frame->scaleFactors_), N_(frame->N_), fuseKFId_(0),
      xMin_(frame->xMin_), xMax_(frame->xMax_), yMin_(frame->yMin_), yMax_(frame->yMax_), 
      gridPerPixelWidth_(frame->gridPerPixelWidth_),
      gridPerPixelHeight_(frame->gridPerPixelHeight_),
      badFlag_(false), voc_(frame->voc_), bowVec_(frame->bowVec_), featVec_(frame->featVec_), 
      relocateFrameId_(0), relocateWordCnt_(0), relocateScore_(0.0f), loopKFId_(0), loopWordCnt_(0), 
      loopScore_(0.0f), notEraseLoopDetecting_(false), toBeEraseAfterLoopDetect_(false),
      localBAKFId_(0), BAFixId_(0), map_(map)
{
    id_ = keyframe_id++;
    
    gridKeypoints_.resize(FRAME_GRID_COLS);
    for (int i = 0; i < FRAME_GRID_COLS; i++)
    {
        gridKeypoints_[i].resize(FRAME_GRID_ROWS);
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            gridKeypoints_[i][j] = frame->gridKeypoints_[i][j];
    }
}

SE3 KeyFrame::getPose()
{
    unique_lock<mutex> lock(mutexPose_);
    return Tcw_;
}

void KeyFrame::setPose(SE3 &Tcw)
{
    unique_lock<mutex> lock(mutexPose_);
    Tcw_ = Tcw;
    Ow_ = Tcw_.inverse().translation();
}

void KeyFrame::addMapPoint(MapPoint* mp, const size_t &idx)
{
    unique_lock<mutex> lock(mutexFeature_);
    mappoints_[idx] = mp;
}

void KeyFrame::replaceMapPoint(MapPoint* mp, const size_t &idx)
{
    mappoints_[idx] = mp;
}

void KeyFrame::setMapPointNull(const size_t &idx)
{
    unique_lock<mutex> lock(mutexFeature_);
    mappoints_[idx] = static_cast<MapPoint*>(nullptr);
}

bool KeyFrame::isInImg(const float &u, const float &v)
{
    return (u>=xMin_ && v>=yMin_ && u<xMax_ && v<yMax_);
}

void KeyFrame::updateConnections()
{
    map<KeyFrame*, int> connections;
    vector<MapPoint*> mappoints;
    
    {
        unique_lock<mutex> lock(mutexFeature_);
        mappoints = mappoints_;
    }
    
    // get observed keyframes of each mappoint
    for (auto itm = mappoints.begin(), itme = mappoints.end(); itm != itme; itm++)
    {
        MapPoint* mp = *itm;
        if (!mp || mp->isBad())
            continue;
        
        map<KeyFrame*, size_t> observedKFs = mp->getObservedKFs();
        for (auto itf = observedKFs.begin(), itfe = observedKFs.end(); itf != itfe; itf++)
        {
            if (itf->first->id_ == id_)
                continue;
            connections[itf->first]++;
        }
    }
    
    if (connections.empty())
        return;
    
    int nmax = 0;
    KeyFrame* kfmax = static_cast<KeyFrame*>(nullptr);
    int threshold = 15;
    
    // common observed mappoints high than threshold consider as connected
    vector< pair<int, KeyFrame*> > pairs;
    pairs.reserve(connections.size());
    for (auto itc = connections.begin(), itce = connections.end(); itc != itce; itc++)
    {
        if (itc->second > nmax)
        {
            nmax = itc->second;
            kfmax = itc->first;
        }
        if (itc->second >= threshold)
        {
            pairs.push_back( make_pair(itc->second, itc->first) );
            (itc->first)->addConnection(this, itc->second); // update covisible frame
        }
    }
    
    // if no one qualified, choose the highest one
    if (pairs.empty())
    {
        pairs.push_back(make_pair(nmax, kfmax));
        kfmax->addConnection(this, nmax);
    }
    
    // sort common points num form low to high, store in list from high to low
    sort(pairs.begin(), pairs.end());
    list<KeyFrame*> kfList;
    list<int> wtList;
    for (size_t i = 0, N = pairs.size(); i < N; i++)
    {
        kfList.push_front(pairs[i].second);
        wtList.push_front(pairs[i].first);
    }
    
    {
        unique_lock<mutex> lock1(mutexConnection_);
        
        // update covisible graph
        connectedKFWts_ = connections;
        orderedConnectKFs_ = vector<KeyFrame*>(kfList.begin(), kfList.end());
        orderedWTs_ = vector<int>(wtList.begin(), wtList.end());
        
        // update spanning tree
        if (firstConnect_ && id_ != 0)
        {
            parent_ = orderedConnectKFs_.front();
            parent_->setChild(this);
            firstConnect_ = false;
        }
    }
}

/**
 * @brief add connected frame or update wt
 */
void KeyFrame::addConnection(KeyFrame* kf, const int& wt)
{
    {
        unique_lock<mutex> lock(mutexConnection_);
        
        if (!connectedKFWts_.count(kf))
            connectedKFWts_[kf] = wt;
        else if (connectedKFWts_[kf] != wt)
            connectedKFWts_[kf] = wt;
        else
            return;
    }
    
    updateBestCovisibles();
}

/**
 * @brief update order after addConnection(initial order at updateConnections) 
 */
void KeyFrame::updateBestCovisibles()
{
    unique_lock<mutex> lock(mutexConnection_);
    
    vector< pair<int, KeyFrame*> > pairs;
    pairs.reserve(connectedKFWts_.size());
    
    for (auto it = connectedKFWts_.begin(), ite = connectedKFWts_.end(); it != ite; it++)
        pairs.push_back(make_pair(it->second, it->first));
    
    sort(pairs.begin(), pairs.end());
    
    list<KeyFrame*> kfList;
    list<int> wtList;
    for (size_t i = 0, N = pairs.size(); i < N; i++)
    {
        kfList.push_front(pairs[i].second);
        wtList.push_front(pairs[i].first);
    }
    
    orderedConnectKFs_ = vector<KeyFrame*>(kfList.begin(), kfList.end());
    orderedWTs_ = vector<int>(wtList.begin(), wtList.end());
}

vector<KeyFrame*> KeyFrame::getBestCovisibleKFs(const int &N)
{
    unique_lock<mutex> lock(mutexConnection_);
    
    if (orderedConnectKFs_.size() < N)
        return orderedConnectKFs_;
    else
        return vector<KeyFrame*>(orderedConnectKFs_.begin(), orderedConnectKFs_.begin()+N);
}

int KeyFrame::trackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mutexFeature_);
    
    int cnt = 0;
    for (int i = 0, N = mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = mappoints_[i];
        if (mp && !mp->isBad())
        {
            if (mp->getObsCnt() >= minObs)
                cnt++;
        }
    }
    
    return cnt;
}

Vector3d KeyFrame::getCamCenter()
{
    unique_lock<mutex> lock(mutexPose_);
    return Ow_;
}

double KeyFrame::computeMidDepth()
{
    vector<MapPoint*> mappoints;
    SE3 Tcw;
    {
        unique_lock<mutex> lock1(mutexFeature_);
        unique_lock<mutex> lock2(mutexPose_);
        
        mappoints = mappoints_;
        Tcw = Tcw_;
    }
    
    vector<double> depths;
    depths.reserve(N_);
    
    Eigen::Matrix3d R = Tcw.rotation_matrix();
    Vector3d R2 = R.row(2).transpose();
    double zcw = Tcw.translation()[2];
    
    for (int i = 0; i < N_; i++)
    {
        MapPoint* mp = mappoints[i];
        if (mp)
        {
            double z = R2.dot(mp->getPose()) + zcw;
            depths.push_back(z);
        }
    }
    
    sort(depths.begin(), depths.end());
    
    return depths[int( (depths.size()-1)/2 )];
}

vector<int> KeyFrame::getFeaturesInArea(const float& u, const float& v,const float& radius)
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
                const float distx = kpt.pt.x - u;
                const float disty = kpt.pt.y - v;
                
                if (fabs(distx) < radius && fabs(disty) < radius)
                    indexs.push_back(keypointIndexs[i]);
            }
        }
    }
    
    return indexs;
}


set<KeyFrame*> KeyFrame::getConnectKFs()
{
    set<KeyFrame*> connectKFs;
    unique_lock<mutex> lock(mutexConnection_);
    
    for (auto it = connectedKFWts_.begin(); it != connectedKFWts_.end(); it++)
        connectKFs.insert(it->first);
    return connectKFs;
}

vector<KeyFrame*> KeyFrame::getOrderedKFs()
{
    unique_lock<mutex> lock(mutexConnection_);
    return orderedConnectKFs_;
}

int KeyFrame::getWeight(KeyFrame* keyframe)
{
    unique_lock<mutex> lock(mutexConnection_);
    
    if (connectedKFWts_.count(keyframe))
        return connectedKFWts_[keyframe];
    else
        return 0;
}

vector<KeyFrame*> KeyFrame::getCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mutexConnection_);
    
    if (orderedConnectKFs_.empty())
        return vector<KeyFrame*>();
    
    vector<int>::iterator it = upper_bound(orderedWTs_.begin(), orderedWTs_.end(), w,
                                           [] (int a, int b) {return a>b;});
    if (it == orderedWTs_.end() && *orderedWTs_.rbegin() < w)
        return vector<KeyFrame*>();
    else
    {
        int n = it - orderedWTs_.begin();
        return vector<KeyFrame*>(orderedConnectKFs_.begin(), orderedConnectKFs_.begin()+n);
    }
        
}

vector<cv::KeyPoint> KeyFrame::getKeyPoints()
{
    unique_lock<mutex> lock(mutexFeature_);
    return unKeypoints_;
}

vector<MapPoint*> KeyFrame::getMapPoints()
{
    unique_lock<mutex> lock(mutexFeature_);
    return mappoints_;
}

set<KeyFrame*> KeyFrame::getChildren()
{
    unique_lock<mutex> lock(mutexFeature_);
    return children_;
}

KeyFrame* KeyFrame::getParent()
{
    unique_lock<mutex> lock(mutexFeature_);
    return parent_;
}

vector<Mat> KeyFrame::getDespVector()
{
    vector<Mat> desps;
    desps.reserve(N_);
    for (int i = 0; i < descriptors_.rows; i++)
        desps.push_back(descriptors_.row(i).clone());
    
    return desps;
}

void KeyFrame::computeBow()
{
    if (bowVec_.empty())
        voc_->transform(getDespVector(), bowVec_, featVec_, 3);
}

void KeyFrame::eraseKeyFrame()
{
    if (id_ == 0)
        return;
    
    {
        unique_lock<mutex> lock(mutexConnection_);
        
        if (notEraseLoopDetecting_)
        {
            toBeEraseAfterLoopDetect_ = true;
            return;
        }
    }
    
    for (auto it = connectedKFWts_.begin(), ite = connectedKFWts_.end(); it != ite; it++)
        it->first->eraseConnection(this);
    
    for (size_t i = 0, N = mappoints_.size(); i < N; i++)
        if (mappoints_[i])
            mappoints_[i]->eraseObservedKF(this);
    
    {
        unique_lock<mutex> lock1(mutexConnection_);
        unique_lock<mutex> lock2(mutexFeature_);
        
        connectedKFWts_.clear();
        orderedConnectKFs_.clear();
        
        set<KeyFrame*> parentCandidates;
        parentCandidates.insert(parent_);
        
        while(!children_.empty())
        {
            bool continueFlag = false;
            
            int weight_max = -1;
            KeyFrame* parentKF;
            KeyFrame* childKF;
            
            for (auto it = children_.begin(), ite = children_.end(); it != ite; it++)
            {
                KeyFrame* kf = *it;
                if (kf->isBad())
                    continue;
                
                // search connection which is also parentCandidates
                vector<KeyFrame*> connections = kf->getOrderedKFs();
                for (auto itc = connections.begin(), itce = connections.end(); itc != itce; itc++)
                {
                    for (auto itp = parentCandidates.begin(), itpe = parentCandidates.end(); itp != itpe; itp++)
                    {
                        if ((*itc)->id_ == (*itp)->id_ )
                        {
                            int weight = kf->getWeight(*itc);
                            
                            if (weight > weight_max)
                            {
                                weight_max = weight;
                                
                                parentKF = *itc;
                                childKF = kf;
                                continueFlag = true;
                            }
                        }
                    }
                }
            }
            
            if (continueFlag)
            {
                childKF->setParent(parentKF);
                parentCandidates.insert(childKF);
                children_.erase(childKF);
            }
            else
                break;
        }
        
        if (!children_.empty())
        {
            for (auto itr = children_.begin(), itre = children_.end(); itr != itre; itr++)
                (*itr)->setParent(parent_);
        }
        
        parent_->eraseChild(this);
        Tcp_ = Tcw_ * parent_->Tcw_.inverse(); // usage: recover trajectory if keyframe was culled
        badFlag_ = true;
    }
    
    map_->eraseKeyFrame(this);
}

void KeyFrame::setParent(KeyFrame* kf)
{
    unique_lock<mutex> lock(mutexConnection_);
    parent_ = kf;
    kf->setChild(this);
}

void KeyFrame::setChild(KeyFrame* kf)
{
    unique_lock<mutex> lock(mutexConnection_);
    children_.insert(kf);
}

void KeyFrame::eraseChild(KeyFrame* kf)
{
    unique_lock<mutex> lock(mutexConnection_);
    children_.erase(kf);
}

void KeyFrame::eraseConnection(KeyFrame* kf)
{
    bool eraseFlag = false;
    {
        unique_lock<mutex> lock(mutexConnection_);
        if (connectedKFWts_.count(kf))
        {
            connectedKFWts_.erase(kf);
            eraseFlag = true;
        }
    }
    
    if (eraseFlag)
        updateBestCovisibles();
}

void KeyFrame::addLoopEdge(KeyFrame* kf)
{
    unique_lock<mutex> lock(mutexConnection_);
    notEraseLoopDetecting_ = true;
    loopEdges_.insert(kf);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mutexConnection_);
    return badFlag_;
}
    
void KeyFrame::setNotEraseLoopDetectingKF()
{
    unique_lock<mutex> lock(mutexConnection_);
    notEraseLoopDetecting_ = true;
}
    
void KeyFrame::setEraseLoopDetectingKF()
{
    {
        unique_lock<mutex> lock(mutexConnection_);
        if(loopEdges_.empty())
            notEraseLoopDetecting_ = false;
    }
    
    if (toBeEraseAfterLoopDetect_)
        eraseKeyFrame();
}
    
}
