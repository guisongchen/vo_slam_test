#include "myslam/loopClosing.h"
#include "myslam/matcher.h"
#include "myslam/optimizer_ceres.h"
#include "myslam/sim3Solver.h"

#include <thread>

#define DEBUG 0

namespace myslam
{
    
LoopClosing::LoopClosing(Map* map)
    : lastLoopKFId_(0) , map_(map), keyframe_match_(nullptr), Scw_(Sophus::Sim3()), 
      finishFlag_(false), requestFinishFlag_(false), fixScaleFlag_(true) {}

void LoopClosing::run()
{
    while (1)
    {
        if (checkNewKeyFrames())
        {
            if (detectLoop())
            {
                if (computeSim3())
                    correctLoop();
            }
        }
        
        if (checkFinishRequest())
            break;
        
        this_thread::sleep_for(chrono::milliseconds(5));
    }
    
    setFinish();
}

bool LoopClosing::checkNewKeyFrames()
{
    unique_lock<mutex> lock(mutexNewKFs_);
    return (!newKeyFrames_.empty());
}

void LoopClosing::insertKeyFrame(KeyFrame* keyframe)
{
    unique_lock<mutex> lock(mutexNewKFs_);
    newKeyFrames_.push_back(keyframe);
}
    

bool LoopClosing::detectLoop()
{
    {
        unique_lock<mutex> lock(mutexNewKFs_);
        keyframe_curr_ = newKeyFrames_.front();
        newKeyFrames_.pop_front();
        
        keyframe_curr_->setNotEraseLoopDetectingKF();
    }
    
    if (keyframe_curr_->id_ < lastLoopKFId_ + 10)
    {
        keyframe_curr_->setEraseLoopDetectingKF();
        return false;
    }
    
    const vector<KeyFrame*> connectKFs = keyframe_curr_->orderedConnectKFs_;
    const DBoW3::BowVector currBowVec = keyframe_curr_->bowVec_;
    
    float minScore = 1.0f;
    for (size_t i = 0; i < connectKFs.size(); i++)
    {
        KeyFrame* kf = connectKFs[i];
        if (kf->isBad())
            continue;
        
        const DBoW3::BowVector bowVec = kf->bowVec_;
        float sc = map_->score(currBowVec, bowVec);
        
        if (sc < minScore)
            minScore = sc;
    }
    
    vector<KeyFrame*> candidates = map_->detectLoopCandidates(keyframe_curr_, minScore);
    
    if (candidates.empty())
    {
        // if candidates empty, the continuity no long exist, clear prev consistent groups
        prevConsistentGroups_.clear();
        keyframe_curr_->setEraseLoopDetectingKF();
        return false;
    }    
    
    enoughConCandidates_.clear();
    
    vector<consistentGroup> currConsistentGroups;
    vector<bool> prevConsistentGroupFlags(prevConsistentGroups_.size(), false); 
    
    // flag to indicate whether prev consistent group is updated by current candidates
    // candidates and prevConsistentGroups both are groups including some keyframes
    // if one from each side is the same, candidates and the prevConsistentGroup is consistent
    // since prevConsistentGroup is backup of prev candidates
    // and keyframes in candidates are consider similar/consistent with each other
    // if one keyframe is the same, we will be confident to say two groups are consistent
    // no need to make sure all keyframes are identical
    
    for (size_t i = 0; i < candidates.size(); i++)
    {
        KeyFrame* kf = candidates[i];
        
        set<KeyFrame*> candidateGroup = kf->getConnectKFs();
        candidateGroup.insert(kf);
        
        bool enoughConsistent = false;
        bool someConsistent = false;
        
        // if prevConsistentGroups_ empty, someConsistent will be false
        // current candidates will be add to currConsistentGroups
        // then backup data in prevConsistentGroups_ for next loop detect
        for (size_t j = 0; j < prevConsistentGroups_.size(); j++)
        {
            // use set to judge if current candidates exist in prev consistent group
            set<KeyFrame*> prevGroup = prevConsistentGroups_[j].first;
            
            bool consist = false;
            for (auto it = candidateGroup.begin(), ite = candidateGroup.end(); it != ite; it++)
            {
                if (prevGroup.count(*it))
                {
                    consist = true;
                    someConsistent = true;
                    break;
                }
            }
            
            if (consist)
            {
                int prevCnt = prevConsistentGroups_[j].second;
                int currCnt = prevCnt + 1;
                
                if (!prevConsistentGroupFlags[j])
                {
                    consistentGroup cg = make_pair(candidateGroup, currCnt);
                    currConsistentGroups.push_back(cg);
                    prevConsistentGroupFlags[j] = true; // make sure only update only once
                }
                
                if (currCnt >= 3 && !enoughConsistent)
                {
                    enoughConCandidates_.push_back(kf);
                    enoughConsistent = true;
                }
            }
        }
        
        // if no match form candidates and prevConsistentGroups
        if (!someConsistent)
        {
            consistentGroup cg = make_pair(candidateGroup, 0);
            currConsistentGroups.push_back(cg);
        }
    }
    
    prevConsistentGroups_ = currConsistentGroups;
    
    if (enoughConCandidates_.empty())
    {
        keyframe_curr_->setEraseLoopDetectingKF();
        return false;
    }
    else
        return true;
    
}


bool LoopClosing::computeSim3()
{
    const int initCnt = enoughConCandidates_.size();
    
    Matcher matcher(0.75);
    
    vector<Sim3Solver*> sim3Solvers;
    sim3Solvers.resize(initCnt);
    
    vector< vector<MapPoint*> > mappointMatches;
    mappointMatches.resize(initCnt);
    
    vector<bool> discards;
    discards.resize(initCnt);
    
    // for visualodometry, mappoints flow between frames, only one group of mappoints exists
    // for loop closure, there two groups of mappoints, one is from current keyframe
    // the other is from early keyframe which was detected as loop closure
    // two groups are connected by descriptor distance under certain threshold
    // sim3 is computed by minimize projection error on both pixel coordinates
    
    // the challenge is how to store two groups in one containtor(say vector)
    // solution: use both index and element. (connect exists when element is NOT null)
    // vector index: index of current keyframe mappoint, get mappoint by index
    // vector element: mappoint of matched keyframe, get correspond index by member function 
    
    int candidates = 0;
    for (int i = 0; i < initCnt; i++)
    {
        KeyFrame* kf = enoughConCandidates_[i];
        kf->setNotEraseLoopDetectingKF();
        
        if (kf->isBad())
        {
            discards[i] = true;
            continue;
        }
        
        int matches = matcher.searchByBoW(keyframe_curr_, kf, mappointMatches[i], true);
        
        if (matches < 20)
        {
            discards[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* solver = new Sim3Solver(keyframe_curr_, kf, mappointMatches[i], fixScaleFlag_);
            
            solver->setRansacParameters(0.99, 20, 300);
            sim3Solvers[i] = solver;
        }
        
        candidates++;
    }
    
    bool matchFlag = false;
    
    while (candidates > 0 && !matchFlag)
    {
        for (int i = 0; i < initCnt; i++)
        {
            if (discards[i])
                continue;
            
            KeyFrame* kf = enoughConCandidates_[i];
            
            vector<bool> inlierFlags;
            int inlier_cnt;
            bool stopFlag;
            bool emptyFlag;
            
            Sim3Solver* solver = sim3Solvers[i];
            Sophus::Sim3 Scm = solver->iterate(5, stopFlag, emptyFlag, inlierFlags, inlier_cnt);
            
            if (stopFlag)
            {
                discards[i] = true;
                candidates--;
            }
            
            if (!emptyFlag)
            {
                vector<MapPoint*> inlierMappoints(mappointMatches[i].size(), 
                                                  static_cast<MapPoint*>(nullptr));
                for (int j = 0, N = inlierFlags.size(); j < N; j++)
                {
                    if (inlierFlags[j])
                        inlierMappoints[j] = mappointMatches[i][j];
                }
                
                matcher.searchBySim3(keyframe_curr_, kf, inlierMappoints, Scm, 7.5);
                
                inlier_cnt = Optimizer::solveLoopSim3(keyframe_curr_, kf, inlierMappoints, 
                                                      Scm, fixScaleFlag_);
                
                if (inlier_cnt >= 20)
                {
                    matchFlag = true;
                    
                    keyframe_match_ = kf;
                    SE3 Tmw = kf->getPose();
                    Sophus::Sim3 Smw(Sophus::ScSO3(Tmw.unit_quaternion()), Tmw.translation());
                    Scw_ = Scm * Smw;
                    
                    matchMapPoints_ = inlierMappoints;
                    break;
                }
            }
        }
    }
    
    if (!matchFlag)
    {
        for (int i = 0; i < initCnt; i++)
            enoughConCandidates_[i]->setEraseLoopDetectingKF();
        keyframe_curr_->setEraseLoopDetectingKF();
        return false;
    }
    
    vector<KeyFrame*> loopConnectKFs = keyframe_match_->orderedConnectKFs_;
    loopConnectKFs.push_back(keyframe_match_);
    loopConnectKFMapPoints_.clear();
    
    for (auto it = loopConnectKFs.begin(), ite = loopConnectKFs.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        vector<MapPoint*> mappoints = kf->getMapPoints();
        
        for (int i = 0, N = mappoints.size(); i < N; i++)
        {
            MapPoint* mp = mappoints[i];
            if (mp)
            {
                if (!mp->isBad() && mp->loopPointIdxOfKF_ != keyframe_curr_->id_)
                {
                    loopConnectKFMapPoints_.push_back(mp);
                    mp->loopPointIdxOfKF_ = keyframe_curr_->id_;
                }
            }
        }
    }
    
    // find more matched mappoints by sim3 search without scale
    matcher.searchByProjection(keyframe_curr_, Scw_, loopConnectKFMapPoints_, matchMapPoints_, 10);
    
    int match_cnt = 0;
    for (int i = 0, N = matchMapPoints_.size(); i < N; i++)
    {
        if (matchMapPoints_[i])
            match_cnt++;
    }
    
    if (match_cnt >= 40)
    {
        for (int i = 0; i < initCnt; i++)
        {
            if (enoughConCandidates_[i] != keyframe_match_)
                enoughConCandidates_[i]->setEraseLoopDetectingKF();
        }
        
        return true;
    }
    else
    {
        for (int i = 0; i < initCnt; i++)
            enoughConCandidates_[i]->setEraseLoopDetectingKF();
        keyframe_curr_->setEraseLoopDetectingKF();
        return false;
    }
}


void LoopClosing::correctLoop()
{
    cout << "loop detected!!" << endl; 
    cout << "corrected frame  id: " << keyframe_curr_->orgFrameId_ 
         << ";  matched frame id: " << keyframe_match_->orgFrameId_ << endl;

    localMapper_->requestStop();
    
    while (!localMapper_->isStopped())
    {
        this_thread::sleep_for(chrono::milliseconds(1));
    }
    
    keyframe_curr_->updateConnections();
    
    // get connectKFs of current keyframe, also including current keyframe
    currConnectKFs_ = keyframe_curr_->orderedConnectKFs_;
    currConnectKFs_.push_back(keyframe_curr_);
    
    // connected keyframes and Sim3, use map to support efficient query and insert
    // NOTE Now we have Scw, how to use this info to correct connected keyframes? use Sic !
    // basically, we get Tic(SE3 from iKF to currKF), then wrapped into Sic with scale 1
    // correctSim3 w.r.t worldframe after corrected by Scw (Sic*Scw)
    // uncorrectSim3 w.r.t worldFrame (Siw, we use this to update mappoints later)
    KeyFrameAndPose correctSim3, uncorrectSim3;
    correctSim3[keyframe_curr_] = Scw_;
    
    SE3 Twc = keyframe_curr_->getPose().inverse();
    
    {
        unique_lock<mutex> lock(map_->mutexMapUpdate_);
        
        // storage sim3 at worldframe(not correct yet, used uncorrect ones to correct mappoints)
        for (auto it = currConnectKFs_.begin(), ite = currConnectKFs_.end(); it != ite; it++)
        {
            KeyFrame* kf = *it;
            SE3 Tiw = kf->getPose();
            
            if (kf != keyframe_curr_)
            {
                SE3 Tic = Tiw * Twc;
                Sophus::Sim3 Sic(Sophus::ScSO3(Tic.unit_quaternion()), Tic.translation());
                correctSim3[kf] = Sic * Scw_;
            }
            
            Sophus::Sim3 Siw(Sophus::ScSO3(Tiw.unit_quaternion()), Tiw.translation());
            uncorrectSim3[kf] = Siw;
        }
        
        // correct mappoints pose according to reference keyframe
        for (KeyFrameAndPose::iterator it = correctSim3.begin(), ite = correctSim3.end(); it != ite; it++)
        {
            KeyFrame* kf = it->first;
            Sophus::Sim3 correctSiw = it->second;
            Sophus::Sim3 correctSwi = correctSiw.inverse();
            Sophus::Sim3 uncorrectSiw = uncorrectSim3[kf];
            
            vector<MapPoint*> mappoints = kf->getMapPoints();
            for (int i = 0, N = mappoints.size(); i < N; i++)
            {
                MapPoint* mp = mappoints[i];
                if (!mp)
                    continue;
                
                if (mp->isBad() || mp->loopCorrectByKF_ == keyframe_curr_->id_)
                    continue;
                
                // project mappoint pose to uncorrect keyframe(where we get the pose), get pixel info 
                // then unprojected pixel info to correct keyframe, get corrected pose
                Vector3d p3dw = mp->getPose();
                Vector3d correctP3dw = correctSwi * (uncorrectSiw*p3dw);
                
                mp->setPose(correctP3dw);
                mp->loopCorrectByKF_ = keyframe_curr_->id_;
                mp->correctReference_ = kf->id_;
                mp->updateNormalAndDepth();
            }
            
            // correct keyframe pose NOW
            double s = correctSiw.scale();
            
            // sim3 rotation_matrix output unscaled rotation, but quaternion is scaled
            SE3 correctTiw(correctSiw.rotation_matrix(), correctSiw.translation()/s);
            
            kf->setPose(correctTiw);
            kf->updateConnections();
        }
        
        for (int i = 0, N = matchMapPoints_.size(); i < N; i++)
        {
            if (matchMapPoints_[i])
            {
                MapPoint* mpLoop = matchMapPoints_[i];
                MapPoint* mpCurr = keyframe_curr_->mappoints_[i];
                
                if (mpCurr)
                    mpCurr->replaceMapPoint(mpLoop);
                else
                {
                    keyframe_curr_->addMapPoint(mpLoop, i);
                    mpLoop->addObservation(keyframe_curr_, i);
                    mpLoop->computeDescriptor();
                }
            }
        }
    }
    
    searchAndFuse(correctSim3);
    
    // connected by loop closure
    map<KeyFrame*, set<KeyFrame*> > loopConnections;
    
    // earse connections which exist before corrected(first and second connections)
    for (auto it = currConnectKFs_.begin(), ite = currConnectKFs_.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        vector<KeyFrame*> neighbors = kf->orderedConnectKFs_;
        
        kf->updateConnections();
        loopConnections[kf] = kf->getConnectKFs();
        
        // erase second connections
        for (auto itn = neighbors.begin(), itne = neighbors.end(); itn != itne; itn++)
            loopConnections[kf].erase(*itn);
        
        // erase first connections
        for (auto itc = currConnectKFs_.begin(), itce = currConnectKFs_.end(); itc != itce; itc++)
            loopConnections[kf].erase(*itc);
    }
    
    Optimizer::solvePoseGraphLoop(map_, keyframe_match_, keyframe_curr_,
                                  uncorrectSim3, correctSim3, loopConnections, fixScaleFlag_);
    
    keyframe_curr_->addLoopEdge(keyframe_match_);
    keyframe_match_->addLoopEdge(keyframe_curr_);
    
    localMapper_->release();
    
    cout << "loop closed!!!" << endl;
    
    lastLoopKFId_ = keyframe_curr_->id_;
    
}


void LoopClosing::searchAndFuse(const KeyFrameAndPose &correctPose)
{
    Matcher matcher(0.8);
    for (auto it = correctPose.begin(), ite = correctPose.end(); it != ite; it++)
    {
        KeyFrame* kf = it->first;
        Sophus::Sim3 Scw = it->second;
        
        vector<MapPoint*> replaceMapPoints(loopConnectKFMapPoints_.size(), static_cast<MapPoint*>(nullptr));
        matcher.fuseByPose(kf, Scw, loopConnectKFMapPoints_, replaceMapPoints, 4);
        
        unique_lock<mutex> lock(map_->mutexMapUpdate_);
        for (int i = 0, N = loopConnectKFMapPoints_.size(); i < N; i++)
        {
            MapPoint* mp = replaceMapPoints[i];
            if (mp)
                mp->replaceMapPoint(loopConnectKFMapPoints_[i]);
        }
    }
    
}

bool LoopClosing::checkFinishRequest()
{
    unique_lock<mutex> lock(mutexFinish_);
    return requestFinishFlag_;
}

void LoopClosing::requestFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    requestFinishFlag_ = true;
}

bool LoopClosing::checkFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    return finishFlag_;
}

void LoopClosing::setFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    finishFlag_ = true;
}

void LoopClosing::setLocalMapper(LocalMapping* localMapper)
{
    localMapper_ = localMapper;
}

void LoopClosing::setVocabulary(DBoW3::Vocabulary* voc)
{
    voc_ = voc;
}
    
}
