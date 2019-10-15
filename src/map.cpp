#include "myslam/map.h"
#include "myslam/config.h"

namespace myslam
{
    
Map::Map() : voc_(nullptr), maxKFId_(0), saveVocabularyFlag_(false) {}
    
void Map::insertKeyFrame(KeyFrame* keyframe)
{
    {
        unique_lock<mutex> lock(mutexMap_);
        keyframes_.insert(keyframe);
    }
    
    if (keyframe->id_ >maxKFId_)
        maxKFId_ = keyframe->id_;
 
    DBoW3::BowVector::const_iterator it, ite;
    for (it = keyframe->bowVec_.begin(), ite = keyframe->bowVec_.end(); it != ite; it++)
        invertIdxs_[it->first].push_back(keyframe);
}

void Map::insertMapPoint(MapPoint* mappoint)
{
    unique_lock<mutex> lock(mutexMap_);
    mappoints_.insert(mappoint);
}

void Map::eraseMapPoint(MapPoint* mappoint)
{
    unique_lock<mutex> lock(mutexMap_);
    
    mappoints_.erase(mappoint);
}

void Map::eraseKeyFrame(KeyFrame* keyframe)
{
    {
        unique_lock<mutex> lock(mutexMap_);
        keyframes_.erase(keyframe);
    }
    
    DBoW3::BowVector::const_iterator it;
    for (it = keyframe->bowVec_.begin(); it != keyframe->bowVec_.end(); it++)
    {
        list<KeyFrame*> kfs = invertIdxs_[it->first];
        
        for (list<KeyFrame*>::iterator itl = kfs.begin(); itl != kfs.end(); itl++)
        {
            if (keyframe == *itl)
            {
                kfs.erase(itl);
                break;
            }
        }
    }
}

void Map::createVocabulary()
{
    if (!saveVocabularyFlag_)
        return;
    
    vector<Mat> descriptors;
    
    if (!keyframes_.empty())
    {
        for (auto itb = keyframes_.begin(), ite = keyframes_.end(); itb != ite; itb++)
        {
            KeyFrame* kf = *itb;
            if (kf->isBad())
                continue;
            
            descriptors.push_back(kf->descriptors_);
        }        
    }
    
    if (!lostFrames_.empty())
    {
        for (auto it = lostFrames_.begin(); it != lostFrames_.end(); it++)
            descriptors.push_back( (*it)->descriptors_);        
    }
    
    if (descriptors.empty())
        return;
    
    cout << "creat DBoW based on keyframes and lost frames...." << endl;
    
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    
    string out_path = Config::get<string>("vocabulary_out");
    vocab.save(out_path);
    
    cout.clear();
    
    cout << "Done..." << endl;
}

vector<KeyFrame*> Map::detectRelocalizationCandidates(Frame* frame)
{
    list<KeyFrame*> sharingWordKFs;
    {
        unique_lock<mutex> lock(mutexMap_);
        for (auto it = frame->bowVec_.begin(), ite = frame->bowVec_.end(); it != ite; it++)
        {
            list<KeyFrame*> kfs = invertIdxs_[it->first];
            
            for (auto itl = kfs.begin(), itle = kfs.end(); itl != itle; itl++)
            {
                KeyFrame* kf = *itl;
                
                if (kf->relocateFrameId_ != frame->id_)
                {
                    kf->relocateWordCnt_ = 0;
                    kf->relocateFrameId_ = frame->id_;
                    sharingWordKFs.push_back(kf);
                }
                kf->relocateWordCnt_++;
            }
        }
    }
    
    if (sharingWordKFs.empty())
        return vector<KeyFrame*>();
    
    int maxCommonWords = 0;
    for ( auto it = sharingWordKFs.begin(), ite = sharingWordKFs.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        if (kf->relocateWordCnt_ > maxCommonWords)
            maxCommonWords = kf->relocateWordCnt_;
    }
    
    int minCommonWords = 0.8 * maxCommonWords;
    list< pair<float, KeyFrame*> > scoredKeyframes;
    
    for ( auto it = sharingWordKFs.begin(), ite = sharingWordKFs.end(); it != ite; it++)
    {
        KeyFrame* kf = *it;
        
        if (kf->relocateWordCnt_ > minCommonWords)
        {
            float sc = score(frame->bowVec_, kf->bowVec_);
            
            kf->relocateScore_ = sc;
            scoredKeyframes.push_back(make_pair(sc, kf));
        }
    }
    
    if (scoredKeyframes.empty())
        return vector<KeyFrame*>();
    
    list< pair<float, KeyFrame*> > groupScoreKeyframes;
    float bestGroupScore = 0.0f;
    
    for (auto it = scoredKeyframes.begin(), ite = scoredKeyframes.end(); it != ite; it++)
    {
        KeyFrame* kf = it->second;
        vector<KeyFrame*> neighbors = kf->getBestCovisibleKFs(10);
        
        float bestScore = it->first;
        float groupScore = bestScore;
        KeyFrame* bestKF = kf;
        
        for (auto itn = neighbors.begin(), itne = neighbors.end(); itn != itne; itn++)
        {
            KeyFrame* kfn = *itn;
            if (kfn->relocateFrameId_ != frame->id_)
                continue;
            
            groupScore += kfn->relocateScore_;
            
            if (kfn->relocateScore_ > bestScore)
            {
                bestKF = kfn;
                bestScore = kfn->relocateScore_;
            }
        }
        
        groupScoreKeyframes.push_back(make_pair(groupScore, bestKF));
        if (groupScore > bestGroupScore)
            bestGroupScore = groupScore;
    }
    
    float minScore = 0.75f * bestGroupScore;
    set<KeyFrame*> addedKFS;
    vector<KeyFrame*> candidates;
    candidates.reserve(groupScoreKeyframes.size());
    for (auto it = groupScoreKeyframes.begin(), ite = groupScoreKeyframes.end(); it != ite; it++)
    {
        const float score = it->first;
        
        if (score > minScore)
        {
            KeyFrame* kf = it->second;
            
            if (!addedKFS.count(kf))
            {
                addedKFS.insert(kf);
                candidates.push_back(kf);
            }
        }
    }
    
    return candidates;
}

vector<KeyFrame*> Map::detectLoopCandidates(KeyFrame* keyframe, float minScore)
{
    set<KeyFrame*> connectKFs = keyframe->getConnectKFs();
    
    // add current keyframe, of course we don't consider oneself as loopclosing match 
    connectKFs.insert(keyframe);
    
    list<KeyFrame*> sharingWordKFs;
    
    {
        unique_lock<mutex> lock(mutexMap_);
        
        for (auto it = keyframe->bowVec_.begin(), ite = keyframe->bowVec_.end(); it != ite; it++)
        {
            // NOTE current frame also exist in bowVec_, needs to be excluded 
            // because we add current keyframe into map before loopclosing thread
            // ORB_SLAM2 use keyframeDataBase to store, and add current keyframe after loopclosing 
            list<KeyFrame*> KFs = invertIdxs_[it->first];
            
            for (auto itl = KFs.begin(), itle = KFs.end(); itl != itle; itl++)
            {
                KeyFrame* kfi = *itl;
                if (kfi->loopKFId_ != keyframe->id_)
                {
                    kfi->loopWordCnt_ = 0;
                    
                    if (!connectKFs.count(kfi))
                    {
                        kfi->loopKFId_ = keyframe->id_;
                        sharingWordKFs.push_back(kfi);
                    }
                }
                kfi->loopWordCnt_++;
            }
        }
    }
    
    if (sharingWordKFs.empty())
        return vector<KeyFrame*>();
    
    list< pair<float, KeyFrame*> > scoredKeyframes;
    
    int maxCommonWords = 0;
    for (auto itl = sharingWordKFs.begin(), itle= sharingWordKFs.end(); itl != itle; itl++)
    {
        KeyFrame* kfi = *itl;
        
        if (kfi->loopWordCnt_ > maxCommonWords)
            maxCommonWords = kfi->loopWordCnt_;
    }
    
    int minCommonWords = 0.8f * maxCommonWords;
    for (auto itl = sharingWordKFs.begin(), itle = sharingWordKFs.end(); itl != itle; itl++)
    {
        KeyFrame* kfi = *itl;
        
        if (kfi->loopWordCnt_ > minCommonWords)
        {
            float sc = score(keyframe->bowVec_, kfi->bowVec_);
            kfi->loopScore_ = sc;
            if (sc >= minScore)
                scoredKeyframes.push_back(make_pair(sc, kfi));
        }
    }
    
    if (scoredKeyframes.empty())
        return vector<KeyFrame*>();
    
    list< pair<float, KeyFrame*> > groupScoreKeyframes;
    float bestGroupScore = minScore;
    
    for (auto itl = scoredKeyframes.begin(), itle = scoredKeyframes.end(); itl != itle; itl++)
    {
        KeyFrame* kfi = itl->second;
        vector<KeyFrame*> neighbors = kfi->getBestCovisibleKFs(10);
        
        float bestScore = itl->first;
        float groupScore = bestScore;
        KeyFrame* bestKF = kfi;
        
        for (auto itn = neighbors.begin(), itne = neighbors.end(); itn != itne; itn++)
        {
            KeyFrame* kfn = *itn;
            
            if (kfn->loopKFId_ == keyframe->id_ && kfn->loopWordCnt_ > minCommonWords)
            {
                groupScore += kfn->loopScore_;
                
                if (kfn->loopScore_ > bestScore)
                {
                    bestKF = kfn;
                    bestScore = kfn->loopScore_;
                }
            }
        }
        
        groupScoreKeyframes.push_back(make_pair(groupScore, bestKF));
     
        if (groupScore > bestGroupScore)
            bestGroupScore = groupScore;
    }
    
    float minScoreKeep = 0.75f * bestGroupScore;
    set<KeyFrame*> addedKFS;
    vector<KeyFrame*> candidates;
    candidates.reserve(groupScoreKeyframes.size());
    
    for (auto it = groupScoreKeyframes.begin(), ite = groupScoreKeyframes.end(); it != ite; it++)
    {
        const float score = it->first;
        
        if (score > minScoreKeep)
        {
            KeyFrame* kf = it->second;
            if (!addedKFS.count(kf))
            {
                addedKFS.insert(kf);
                candidates.push_back(kf);
            }
        }
    }
    
    return candidates;
}

double Map::score(const DBoW3::BowVector &v1, const DBoW3::BowVector &v2) const
{
    DBoW3::BowVector::const_iterator v1_it, v2_it;
    const DBoW3::BowVector::const_iterator v1_end = v1.end();
    const DBoW3::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW3::WordValue& vi = v1_it->second;
    const DBoW3::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += fabs(vi - wi) - fabs(vi) - fabs(wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  score = -score/2.0;

  return score; // [0..1]
}

void Map::setLocalMapPoints(const vector<MapPoint*> mappoints)
{
    unique_lock<mutex> lock(mutexMap_);
    localMapPoints_ = mappoints;
}

vector<MapPoint*> Map::getLocalMapPoints()
{
    unique_lock<mutex> lock(mutexMap_);
    return localMapPoints_;
}

vector<KeyFrame*> Map::getAllKeyFrames()
{
    unique_lock<mutex> lock(mutexMap_);
    return vector<KeyFrame*>(keyframes_.begin(), keyframes_.end());
}

unsigned long Map::getAllKeyFramesCnt()
{
    unique_lock<mutex> lock(mutexMap_);
    return keyframes_.size();
}

unsigned long Map::getAllMapPointsCnt()
{
    unique_lock<mutex> lock(mutexMap_);
    return mappoints_.size();
}

vector<MapPoint*> Map::getAllMapPoints()
{
    unique_lock<mutex> lock(mutexMap_);
    
    return vector<MapPoint*>(mappoints_.begin(), mappoints_.end());
}
    
}
