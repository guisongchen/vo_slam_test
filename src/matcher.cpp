#include "myslam/matcher.h"
#include "myslam/mappoint.h"
#include <opencv2/features2d.hpp>


#define DEBUG 0

namespace myslam
{
    
const int TH_HIGH = 100;
const int TH_LOW = 50; 
const int HISTO_LENGTH = 30;
const float pdf = HISTO_LENGTH / 360.0f;   // divide 360 into 30 parts
    
Matcher::Matcher(float ratio) : ratio_(ratio) {}

int Matcher::searchByProjection(Frame* frame_curr, Frame* frame_last, const float radius, bool checkRot)
{
    int match_cnt = 0;
    
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    Camera* camera = frame_curr->camera_;
    const int xMax = frame_curr->xMax_;
    const int xMin = frame_curr->xMin_;
    const int yMax = frame_curr->yMax_;
    const int yMin = frame_curr->yMin_;
    const float b = camera->b_;
    const float bf = camera->bf_;
    const int totalLevel = frame_curr->scaleFactors_.size();
    
    SE3 Tcw = frame_curr->Tcw_;
    SE3 Tlc = frame_last->Tcw_ * Tcw.inverse();
    Vector3d tlc = Tlc.translation();
    
    const bool forward = static_cast<float>(tlc[2]) > b;
    const bool backward = -static_cast<float>(tlc[2]) > b;
    
    for (int i = 0, N = frame_last->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = frame_last->mappoints_[i];
        
        if (!mp || frame_last->outliers_[i])
            continue;

        Vector3d p_camera = Tcw * mp->getPose();
        const float z = static_cast<float>(p_camera[2]);
        if (z < 0.0f)
            continue;
        
        const float invz = 1.0f/z;
        Vector2d pixel = camera->camera2pixel(p_camera);
        
        const float u = pixel[0];
        const float v = pixel[1];
        
        if (u<xMin || u>xMax)
            continue;
        if (v<yMin || v>yMax)
            continue;
        

        int lastOctave = frame_last->unKeypoints_[i].octave;
        float radius_scale = radius * frame_curr->scaleFactors_[lastOctave];
        
        vector<int> indexs;
        if(forward)
            indexs = frame_curr->getFeaturesInArea(u, v, radius_scale, lastOctave, totalLevel);
        else if(backward)
            indexs = frame_curr->getFeaturesInArea(u, v, radius_scale, 0, lastOctave);
        else
            indexs = frame_curr->getFeaturesInArea(u, v, radius_scale, lastOctave-1, lastOctave+1);
        
        if (indexs.empty())
            continue;
        
        int bestDist = 256;
        int bestIdx = -1;
        
        const Mat desp_last = mp->getDescriptor();
        for (int j = 0, N = indexs.size(); j < N; j++)
        {
            int idx = indexs[j];
            if (frame_curr->mappoints_[idx] && (frame_curr->mappoints_[idx]->observe_cnt_ > 0) )
                continue;
            
            if (frame_curr->uRight_[idx] > 0)
            {
                const float u_r = u - bf * invz;
                const float error = fabs(u_r - frame_curr->uRight_[idx]);
                if (error > radius_scale)
                    continue;                    
            }
            
            const Mat desp_curr = frame_curr->descriptors_.row(idx);
            const int dist = computeDistance(desp_last, desp_curr);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
            
        if (bestDist <= TH_HIGH)
        {
            frame_curr->mappoints_[bestIdx] = mp;
            match_cnt++;
            
            if (checkRot)
            {
                float rot = frame_last->unKeypoints_[i].angle - frame_curr->unKeypoints_[bestIdx].angle;
                if (rot < 0)
                    rot += 360.0f;
                int distribution = cvRound(rot * pdf);
                if (distribution == HISTO_LENGTH)
                    distribution = 0;
                assert(distribution >= 0 && distribution < HISTO_LENGTH);
                rotHist[distribution].push_back(bestIdx);                    
            }

        }
    }
    
    if (checkRot)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        
        computeThreeMax(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
                for (int j = 0; j < rotHist[i].size(); j++)
                {
                    frame_curr->mappoints_[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
                    match_cnt--;
                }
        }
    }
    
    return match_cnt;
}

int Matcher::searchByProjection(Frame* frame_curr, KeyFrame* keyframe, const float radius,
                                const float distThreshold, const set<MapPoint*> &found, bool checkRot)
{
    int match_cnt = 0;
    
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    Camera* camera = frame_curr->camera_;
    const int xMax = frame_curr->xMax_;
    const int xMin = frame_curr->xMin_;
    const int yMax = frame_curr->yMax_;
    const int yMin = frame_curr->yMin_;
    
    const SE3 Tcw = frame_curr->Tcw_;
    Vector3d Ow = Tcw.inverse().translation();
    
    const vector<MapPoint*> mappoints = keyframe->getMapPoints();
    for (int i = 0, N = mappoints.size(); i < N; i++)
    {
        MapPoint* mp = mappoints[i];
        
        if (!mp)
            continue;
        
        if (mp->isBad() || found.count(mp))
            continue;

        Vector3d p_camera = Tcw * mp->getPose();
        const float z = p_camera[2];
        if (z <= 0)
            continue;

        Vector2d pixel = camera->camera2pixel(p_camera);
        const float &u = pixel[0];
        const float &v = pixel[1];
        
        if (u > xMax || u < xMin)
            continue;
        if (v > yMax || v < yMin)
            continue;
        
        Vector3d line = mp->getPose() - Ow;
        const float distance = line.norm();
        const float maxDistance = mp->getMaxDistanceThreshold();
        const float minDistance = mp->getMinDistanceThreshold(); 
        
        if (distance < minDistance || distance > maxDistance)
            continue;
        
        // two frames maybe not consistent, use predict level instead of keypoint's octave
        int level_predict = mp->predictScale(distance, frame_curr);
        float radius_scale = radius * keyframe->scaleFactors_[level_predict];
        
        vector<int> indexs = frame_curr->getFeaturesInArea(u, v, radius_scale,
                                                           level_predict-1, level_predict+1);
        
        if (indexs.empty())
            continue;
        
        int bestDist = 256;
        int bestIdx = -1;
        
        const Mat desp_last = mp->getDescriptor();
        for (int j = 0, N = indexs.size(); j < N; j++)
        {
            int idx = indexs[j];
            if (frame_curr->mappoints_[idx])
                continue;
            
            const Mat desp_curr = frame_curr->descriptors_.row(idx).clone();
            const int dist = computeDistance(desp_last, desp_curr);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= distThreshold)
        {
            frame_curr->mappoints_[bestIdx] = mp;
            match_cnt++;
            
            if (checkRot)
            {
                float rot = keyframe->unKeypoints_[i].angle - frame_curr->unKeypoints_[bestIdx].angle;
                if (rot < 0)
                    rot += 360.0f;
                int distribution = cvRound(rot * pdf);
                if (distribution == HISTO_LENGTH)
                    distribution = 0;
                assert(distribution >= 0 && distribution < HISTO_LENGTH);
                rotHist[distribution].push_back(bestIdx);                    
            }

        }
    }
    
    if (checkRot)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        
        computeThreeMax(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (int j = 0; j < rotHist[i].size(); j++)
                {
                    frame_curr->mappoints_[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
                    match_cnt--;
                }
            }
        }
    }
    return match_cnt;     
}

int Matcher::searchByProjection(Frame *frame, const vector<MapPoint*> &mappoints, const float thRadius)
{    
    int match_cnt = 0;
    
    for (int im = 0; im < mappoints.size(); im++)
    {
        MapPoint* mp = mappoints[im];
        if (mp->isBad())
            continue;
        
        if (!mp->trackInLocalMap_)
            continue;
        
        float radius;
        if (mp->viewCos_ > 0.998)
            radius = 2.5;
        else
            radius = 4.0;
        
        radius *= thRadius;

        const int level_predict = mp->trackScaleLevel_;
        float radius_scale = radius * frame->scaleFactors_[level_predict];
        const vector<int> indexs = frame->getFeaturesInArea(mp->trackProj_u_, mp->trackProj_v_, radius_scale,
                                                            level_predict-1, level_predict);
        if (indexs.empty())
            continue;
        
        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1;
        
        const Mat desp_local = mp->getDescriptor();
        for (int j = 0; j < indexs.size(); j++)
        {
            const int idx = indexs[j];
            
            // those already exist mappoints form map(obs>0), skip
            if (frame->mappoints_[idx] && frame->mappoints_[idx]->getObsCnt() > 0)
                continue;
            
            if (frame->uRight_[idx] > 0)
            {
                const float er = fabs(mp->trackProj_uR_ - frame->uRight_[idx]);
                if (er > radius_scale)
                    continue;
            }
            
            const Mat &desp_curr = frame->descriptors_.row(idx);
            const int dist = computeDistance(desp_local, desp_curr);
            
            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = frame->unKeypoints_[idx].octave;
                bestIdx = idx;
            }
            else if(dist < bestDist2)
            {
                bestLevel2 = frame->unKeypoints_[idx].octave;
                bestDist2 = dist;
            }
        }
        
        if (bestDist <= TH_HIGH)
        {
            if(bestLevel == bestLevel2 && float(bestDist) > ratio_*float(bestDist2))
                continue;
            
            frame->mappoints_[bestIdx] = mp;
            match_cnt++;
        }
    }
    
    return match_cnt;
}

// search without scale
int Matcher::searchByProjection(KeyFrame* keyframe, Sophus::Sim3 &Scw, vector<MapPoint*> &loopMapPoints,
                                vector<MapPoint*> &matchMapPoints, int th)
{
    Camera* camera = keyframe->camera_;
    const float fx = camera->fx_;
    const float fy = camera->fy_;
    const float cx = camera->cx_;
    const float cy = camera->cy_;
    
    const double scale = Scw.scale();
    Eigen::Matrix3d Rcw = Scw.rotation_matrix() / scale;
    Vector3d tcw = Scw.translation() / scale;
    Vector3d Ow = -Rcw.transpose() * tcw;
    
    set<MapPoint*> alreadyFound(matchMapPoints.begin(), matchMapPoints.end());
    alreadyFound.erase(static_cast<MapPoint*>(nullptr));
    
    int inlier_cnt = 0;
    for (int i = 0, N = loopMapPoints.size(); i < N; i++)
    {
        MapPoint* mp = loopMapPoints[i];
        if (!mp || mp->isBad() || alreadyFound.count(mp))
            continue;
        
        Vector3d p_cam = Rcw * mp->getPose() + tcw;
        const float z = static_cast<float>(p_cam[2]);
        if (z < 0)
            continue;
        
        const float invz = 1.0f/z;
        const float x = static_cast<float>(p_cam[0]) * invz;
        const float y = static_cast<float>(p_cam[1]) * invz;
        
        const float u = fx*x + cx;
        const float v = fy*y + cy;
        
        if (!keyframe->isInImg(u, v))
            continue;
        
        const float max_distance = mp->getMaxDistanceThreshold();
        const float min_distance = mp->getMinDistanceThreshold();
        Vector3d pline = mp->getPose() - Ow;
        const float distance = pline.norm();
        
        if (distance < min_distance || distance > max_distance)
            continue;
        
        Vector3d pNormal = mp->getNormalVector();
        if (pline.dot(pNormal) < 0.5*distance)
            continue;
        
        int level_predict = mp->predictScale(distance, keyframe);
        float radius = th*keyframe->scaleFactors_[level_predict];
        
        const vector<int> indexs = keyframe->getFeaturesInArea(u, v, radius);
        
        if (indexs.empty())
            continue;
        
        Mat desp_mp = mp->getDescriptor();
        int bestDist = 256;
        int bestIdx = -1;
        
        for (int j = 0, N = indexs.size(); j < N; j++)
        {
            const int idx = indexs[j];
            if (matchMapPoints[j])
                continue;
            
            const int level = keyframe->unKeypoints_[idx].octave;
            if (level < level_predict - 1 || level > level_predict)
                continue;
            
            Mat desp_kf = keyframe->descriptors_.row(idx);
            const int dist = computeDistance(desp_mp, desp_kf);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= TH_LOW)
        {
            matchMapPoints[bestIdx] = mp;
            inlier_cnt++;
        }
    }
    
    return inlier_cnt;
}

int Matcher::searchByBoW(KeyFrame* keyframe, Frame* frame, vector<MapPoint*> &mappointMatches, bool checkRot)
{
    int match_cnt = 0;
    
    mappointMatches = vector<MapPoint*>(frame->N_, static_cast<MapPoint*>(nullptr));
    vector<MapPoint*> mappoints = keyframe->getMapPoints();
    
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    DBoW3::FeatureVector::const_iterator kit = keyframe->featVec_.begin();
    DBoW3::FeatureVector::const_iterator kite = keyframe->featVec_.end();
    DBoW3::FeatureVector::const_iterator it = frame->featVec_.begin();
    DBoW3::FeatureVector::const_iterator ite = frame->featVec_.end();
    
    while (kit != kite && it != ite)
    {
        if (kit->first == it->first)
        {
            const vector<unsigned int> kfIndexs = kit->second;
            const vector<unsigned int> fIndexs = it->second;
            
            for (size_t ik = 0; ik < kfIndexs.size(); ik++)
            {
                const unsigned int kfIdx = kfIndexs[ik];
                MapPoint* mpk = mappoints[kfIdx];
                if (!mpk || mpk->isBad())
                    continue;
                
                const Mat kfDesp = keyframe->descriptors_.row(kfIdx);
                
                int bestDist1 = 256;
                int bestIdx = -1;
                int bestDist2 = 256;
                
                for (size_t ir = 0; ir < fIndexs.size(); ir++)
                {
                    const unsigned int fIdx = fIndexs[ir];
                    if (mappointMatches[fIdx])
                        continue;
                    
                    const Mat fDesp = frame->descriptors_.row(fIdx);
                    const int dist = computeDistance(kfDesp, fDesp);
                    
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx = fIdx;
                    }
                    else if (dist < bestDist2)
                        bestDist2 = dist;
                }
                
                if (bestDist1 <= TH_LOW)
                {
                    if ( static_cast<float>(bestDist1) < ratio_*static_cast<float>(bestDist2) )
                    {
                        mappointMatches[bestIdx] = mpk;
                        const cv::KeyPoint kp = keyframe->unKeypoints_[kfIdx];
                        
                        if (checkRot)
                        {
                            float rot = kp.angle - frame->unKeypoints_[bestIdx].angle;
                            if (rot < 0)
                                rot += 360.0f;
                            
                            int distribution = cvRound(rot*pdf);
                            if (distribution == HISTO_LENGTH)
                                distribution = 0;
                            assert(distribution >= 0 && distribution < HISTO_LENGTH);
                            rotHist[distribution].push_back(bestIdx);
                        }
                        match_cnt++;
                    }
                }
            }
            kit++;
            it++;
        }
        
        else if (kit->first < it->first)
            kit = keyframe->featVec_.lower_bound(it->first);
        else
            it = frame->featVec_.lower_bound(kit->first);
    }
    
    if (checkRot)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        
        computeThreeMax(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i==ind1 || i==ind2 || i==ind3)
                continue;
            
            for (int j = 0; j < rotHist[i].size(); j++)
            {
                mappointMatches[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
                match_cnt--;
            }
        }
    }
    
    return match_cnt;
}

int Matcher::searchByBoW(KeyFrame* keyframe1, KeyFrame* keyframe2, vector<MapPoint*> &mappointMatches, bool checkRot)
{
    int match_cnt = 0;

    mappointMatches = vector<MapPoint*>(keyframe1->N_, static_cast<MapPoint*>(nullptr));
    vector<bool> matched2(keyframe2->N_, false);
    
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    
    const vector<MapPoint*> mappoints1 = keyframe1->getMapPoints();
    const vector<MapPoint*> mappoints2 = keyframe2->getMapPoints();

    DBoW3::FeatureVector::const_iterator kit1 = keyframe1->featVec_.begin();
    DBoW3::FeatureVector::const_iterator kit1e = keyframe1->featVec_.end();
    DBoW3::FeatureVector::const_iterator kit2 = keyframe2->featVec_.begin();
    DBoW3::FeatureVector::const_iterator kit2e = keyframe2->featVec_.end();
    
    while (kit1 != kit1e && kit2 != kit2e)
    {
        if (kit1->first == kit2->first)
        {
            const vector<unsigned int> kf1Indexs = kit1->second;
            const vector<unsigned int> kf2Indexs = kit2->second;
            
            for (size_t ik = 0, ike = kit1->second.size(); ik < ike; ik++)
            {
                const unsigned int idx1 = kf1Indexs[ik];
                MapPoint* mpk = mappoints1[idx1];
                if (!mpk || mpk->isBad())
                    continue;
                
                const Mat desp1 = keyframe1->descriptors_.row(idx1);
                
                int bestDist1 = 256;
                int bestIdx2 = -1;
                int bestDist2 = 256;
                
                for (size_t ir = 0, ire = kf2Indexs.size(); ir < ire; ir++)
                {
                    const unsigned int idx2 = kf2Indexs[ir];
                    MapPoint* mpr = mappoints2[idx2];
                    
                    if (matched2[idx2]  || !mpr)
                        continue;
                    if (mpr->isBad())
                        continue;
                    
                    const Mat desp2 = keyframe2->descriptors_.row(idx2);
                    const int dist = computeDistance(desp1, desp2);
                    
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = static_cast<int>(idx2);
                    }
                    else if (dist < bestDist2)
                        bestDist2 = dist;
                }
                
                if (bestDist1 <= TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < ratio_* static_cast<float>(bestDist2))
                    {
                        mappointMatches[idx1] = mappoints2[bestIdx2];
                        matched2[bestIdx2] = true;
                        
                        if (checkRot)
                        {
                            float rot = keyframe1->unKeypoints_[idx1].angle - keyframe2->unKeypoints_[bestIdx2].angle;
                            if (rot < 0)
                                rot += 360.0f;
                            
                            int distribution = round(rot*pdf);
                            if (distribution == HISTO_LENGTH)
                                distribution = 0;
                            assert(distribution >= 0 && distribution < HISTO_LENGTH);
                            rotHist[distribution].push_back(static_cast<int>(idx1));
                        }
                        match_cnt++;
                    }
                }
            }
            kit1++;
            kit2++;
        }
        
        else if (kit1->first < kit2->first)
            kit1 = keyframe1->featVec_.lower_bound(kit2->first);
        else
            kit2 = keyframe2->featVec_.lower_bound(kit1->first);
    }
    
    if (checkRot)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        
        computeThreeMax(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
                for (int j = 0; j < rotHist[i].size(); j++)
                {
                    mappointMatches[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
                    match_cnt--;
                }
        }
    }
    
    return match_cnt;
}

int Matcher::searchBySim3(KeyFrame* keyframe1, KeyFrame* keyframe2, vector<MapPoint*> &matches12, 
                          Sophus::Sim3 &S12, const float th)
{
    const float fx = keyframe1->camera_->fx_;
    const float fy = keyframe1->camera_->fy_;
    const float cx = keyframe1->camera_->cx_;
    const float cy = keyframe1->camera_->cy_;
    
    vector<MapPoint*> mappoints1 = keyframe1->getMapPoints();
    vector<MapPoint*> mappoints2 = keyframe2->getMapPoints();
    const int N1 = mappoints1.size();
    const int N2 = mappoints2.size();
    
    vector<bool> matched1(N1, false);
    vector<bool> matched2(N2, false);
    
    SE3 Tcw1 = keyframe1->getPose();
    SE3 Tcw2 = keyframe2->getPose();
    Sophus::Sim3 S21 = S12.inverse();
    
    for (int i = 0; i < N1; i++)
    {
        MapPoint* mp = matches12[i];
        
        if (mp)
        {
            matched1[i] = true;
            int idx2 = mp->getIndexInKeyFrame(keyframe2);
            if (idx2 >= 0 && idx2 < N2)
                matched2[idx2] = true; 
        }
    }
    
    vector<int> match1(N1, -1);
    vector<int> match2(N2, -1);
    
    for (int i = 0; i < N1; i++)
    {
        MapPoint* mp = mappoints1[i];
        if (!mp || matched1[i])
            continue;
        
        if (mp->isBad())
            continue;
        
        Vector3d p_world = mp->pos_;
        Vector3d p_camera1 = Tcw1 * p_world;
        Vector3d p_camera2 = S21 * p_camera1;
        const float z = static_cast<float>(p_camera2[2]);
        
        if ( z < 0)
            continue;
        
        const float invz = 1.0f / z;
        const float x = static_cast<float>(p_camera2[0]) * invz;
        const float y = static_cast<float>(p_camera2[1]) * invz;
        
        const float u = fx * x + cx;
        const float v = fy * y + cy;
        
        if (!keyframe2->isInImg(u, v))
            continue;
        
        const float maxDistance_ = mp->getMaxDistanceThreshold();
        const float minDistance_ = mp->getMinDistanceThreshold();
        const float distance = p_camera2.norm();
        
        if (distance < minDistance_ || distance > maxDistance_)
            continue;
        
        const int level_predict = mp->predictScale(distance, keyframe2);
        const float radius = th * keyframe2->scaleFactors_[level_predict];
        
        const vector<int> indexs = keyframe2->getFeaturesInArea(u, v, radius);
        if (indexs.empty())
            continue;
        
        const Mat desp1 = mp->getDescriptor();
        
        int bestDist = 256;
        int bestIdx = -1;
        
        for (auto it = indexs.begin(); it != indexs.end(); it++)
        {
            const int idx = *it;
            const cv::KeyPoint kp = keyframe2->unKeypoints_[idx];
            
            if (kp.octave < level_predict - 1 || kp.octave > level_predict)
                continue;
            
            const Mat desp2 = keyframe2->descriptors_.row(idx);
            const int dist = computeDistance(desp1, desp2);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= TH_HIGH)
            match1[i] = bestIdx;
    }
    
    for (int j = 0; j < N2; j++)
    {
        MapPoint* mp = mappoints2[j];
        if (!mp || matched2[j])
            continue;
        
        Vector3d p_world = mp->pos_;
        Vector3d p_camera2 = Tcw2 * p_world;
        Vector3d p_camera1 = S12 * p_camera2;
        const float z = static_cast<float>(p_camera1[2]);
        
        if ( z <= 0)
            continue;
        
        const float invz = 1.0f / z;
        const float x = static_cast<float>(p_camera1[0]) * invz;
        const float y = static_cast<float>(p_camera1[1]) * invz;
        
        const float u = fx * x + cx;
        const float v = fy * y + cy;
        
        if (!keyframe1->isInImg(u, v))
            continue;
        
        const float maxDistance_ = mp->getMaxDistanceThreshold();
        const float minDistance_ = mp->getMinDistanceThreshold();
        const float distance = p_camera1.norm();
        
        if (distance < minDistance_ || distance > maxDistance_)
            continue;
        
        const int level_predict = mp->predictScale(distance, keyframe1);
        const float radius = th * keyframe1->scaleFactors_[level_predict];
        
        const vector<int> indexs = keyframe1->getFeaturesInArea(u, v, radius);
        if (indexs.empty())
            continue;
        
        const Mat desp2 = mp->getDescriptor();
        
        int bestDist = 256;
        int bestIdx = -1;
        
        for (auto it = indexs.begin(); it != indexs.end(); it++)
        {
            const int idx = *it;
            const cv::KeyPoint kp = keyframe1->unKeypoints_[idx];
            
            if (kp.octave < level_predict - 1 || kp.octave > level_predict)
                continue;
            
            const Mat desp1 = keyframe1->descriptors_.row(idx);
            const int dist = computeDistance(desp1, desp2);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= TH_HIGH)
            match2[j] = bestIdx;
    }
    
    int found = 0;
    for (int i = 0; i < N1; i++)
    {
        int idx2 = match1[i];
        if (idx2 >= 0)
        {
            int idx1 = match2[idx2];
            if (idx1 == i)
            {
                matches12[i] = mappoints2[idx2];
                found++;
            }
        }
    }
    
    return found;
    
}

int Matcher::searchForTriangulation(KeyFrame *keyframe1, KeyFrame *keyframe2,
                                    vector< pair<int, int> > &matchIdxs, Eigen::Matrix3d &F12, bool checkRot)
{   
    int match_cnt = 0;

    vector<int> matches12(keyframe1->N_, -1);
    vector<bool> matched2(keyframe2->N_, false);
    
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    const vector<MapPoint*> mappoints1 = keyframe1->getMapPoints();
    const vector<MapPoint*> mappoints2 = keyframe2->getMapPoints();

    DBoW3::FeatureVector::const_iterator kit1 = keyframe1->featVec_.begin();
    DBoW3::FeatureVector::const_iterator kit1e = keyframe1->featVec_.end();
    DBoW3::FeatureVector::const_iterator kit2 = keyframe2->featVec_.begin();
    DBoW3::FeatureVector::const_iterator kit2e = keyframe2->featVec_.end();
    
    const Vector3d Cw = keyframe1->getCamCenter();
    const Vector3d C2 = keyframe2->getPose() * Cw;
    const Vector2d C2_pixel = keyframe2->camera_->camera2pixel(C2);
    const float ex = C2_pixel[0];
    const float ey = C2_pixel[1];
    
    while (kit1 != kit1e && kit2 != kit2e)
    {
        if (kit1->first == kit2->first)
        {
            for (size_t ik = 0, ike = kit1->second.size(); ik < ike; ik++)
            {
                const unsigned int idx1 = kit1->second[ik];
                MapPoint* mpk = mappoints1[idx1];
                
                // looking for mappoints not exist
                if (mpk)
                    continue;
                
                const bool stereo1 = keyframe1->uRight_[idx1] >= 0;
                
                const cv::KeyPoint kpt1 = keyframe1->unKeypoints_[idx1];
                const Mat desp1 = keyframe1->descriptors_.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for (size_t ir = 0, ire = kit2->second.size(); ir < ire; ir++)
                {
                    const unsigned int idx2 = kit2->second[ir];
                    MapPoint* mpr = mappoints2[idx2];
                    
                    // looking for mappoints not exist or not matched
                    if (matched2[idx2] || mpr)
                        continue;
                    
                    const bool stereo2 = keyframe2->uRight_[idx2] >= 0;
                    const Mat desp2 = keyframe2->descriptors_.row(idx2);
                    
                    const int dist = computeDistance(desp1, desp2);
                    
                    if (dist>TH_LOW || dist>bestDist)
                        continue;
                    
                    const cv::KeyPoint kpt2 = keyframe2->unKeypoints_[idx2];
                    if (!stereo1 && !stereo2)
                    {
                        const float distex = ex - kpt2.pt.x;
                        const float distey = ey - kpt2.pt.y;
                        
                        // if keypoint is close to epipoles, means it is too close to camera 1
                        if (distex*distex + distey*distey < 100*keyframe2->scaleFactors_[kpt2.octave])
                            continue;
                    }
                    
                    if (checkEpipolarConstrain(kpt1, kpt2, F12, keyframe2))
                    {
                        bestDist = dist;
                        bestIdx2 = static_cast<int>(idx2);
                    }
                    
                }
                
                if (bestIdx2 >= 0)
                {
                    matches12[idx1] = bestIdx2;
                    matched2[bestIdx2] = true;
                        
                    if (checkRot)
                    {
                        float rot = keyframe1->unKeypoints_[idx1].angle - keyframe2->unKeypoints_[bestIdx2].angle;
                        if (rot < 0)
                            rot += 360.0f;
                        
                        int distribution = round(rot*pdf);
                        if (distribution == HISTO_LENGTH)
                            distribution = 0;
                        assert(distribution >= 0 && distribution < HISTO_LENGTH);
                        rotHist[distribution].push_back(static_cast<int>(idx1));
                    }
                    match_cnt++;
                }
            }
            kit1++;
            kit2++;
        }
        
        else if (kit1->first < kit2->first)
            kit1 = keyframe1->featVec_.lower_bound(kit2->first);
        else
            kit2 = keyframe2->featVec_.lower_bound(kit1->first);
    }
    
    if (checkRot)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        
        computeThreeMax(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
                for (int j = 0; j < rotHist[i].size(); j++)
                {
                    matches12[rotHist[i][j]] = -1;
                    match_cnt--;
                }
        }
    }
    
    matchIdxs.clear();
    matchIdxs.reserve(match_cnt);
    
    for (int i = 0, N = matches12.size(); i < N; i++)
    {
        if (matches12[i] < 0)
            continue;
        matchIdxs.push_back(make_pair(i, matches12[i]));
    }
    
    return match_cnt;
}

int Matcher::fuseMapPoints(KeyFrame* keyframe, vector<MapPoint*> &mappoints, const float &threshold)
{
    int cnt = 0;
    
    Camera* camera = keyframe->camera_;
    const float fx = camera->fx_;
    const float fy = camera->fy_;
    const float cx = camera->cx_;
    const float cy = camera->cy_;
    const float bf = camera->bf_;
    
    SE3 Tcw = keyframe->getPose();
    Vector3d Ow = keyframe->getCamCenter();
    
    for (int i = 0, N = mappoints.size(); i < N; i++)
    {
        MapPoint* mp = mappoints[i];
        if (!mp || mp->isBad() || mp->beObserved(keyframe))
            continue;
        
        Vector3d p_world = mp->getPose();
        Vector3d pcam = Tcw * p_world;
        const float z = static_cast<float>(pcam[2]);
        
        if (z < 0.0f)
            continue;
        
        const float invz = 1.0f/z;
        const float x = static_cast<float>(pcam[0]) * invz;
        const float y = static_cast<float>(pcam[1]) * invz;
        const float u = fx*x + cx;
        const float v = fy*y + cy;
        
        if (!keyframe->isInImg(u,v))
            continue;
        
        const float ur = u - bf*invz;
        
        Vector3d line = p_world - Ow;
        const float dist = line.norm();
        const float minDistance = mp->getMinDistanceThreshold();
        const float maxDistance = mp->getMaxDistanceThreshold();
        
        if (dist < minDistance || dist > maxDistance)
            continue;
        
        Vector3d pn = mp->getNormalVector();
        if (line.dot(pn) < 0.5*dist)
            continue;
        
        int level_predict = mp->predictScale(dist,keyframe);
        const float radius = threshold * keyframe->scaleFactors_[level_predict];
        const vector<int> indexs = keyframe->getFeaturesInArea(u, v, radius);
        if (indexs.empty())
            continue;
        
        int bestDist = 256;
        int bestIdx = -1;
        
        const Mat desp = mp->getDescriptor();
        for (auto it = indexs.begin(), ite = indexs.end(); it != ite; it++)
        {
            const int idx = *it;
            const cv::KeyPoint kp = keyframe->unKeypoints_[idx];
            
            if (kp.octave < (level_predict-1) || kp.octave > level_predict)
                continue;

            const float ex = u - kp.pt.x;
            const float ey = v - kp.pt.y;
            const float invSigma = 1.0f / keyframe->scaleFactors_[kp.octave];
            
            if (keyframe->uRight_[idx] >= 0)
            {
                const float kpr = keyframe->uRight_[idx];
                const float er = ur - kpr;
                const float e2 = ex*ex+ey*ey+er*er;
                
                if (e2*invSigma*invSigma > 7.815f)
                    continue;
            }
            else
            {
                const float e2 = ex*ex+ey*ey;
                
                if (e2*invSigma*invSigma > 5.991f)
                    continue;
            }
            
            const int distance = computeDistance(desp, keyframe->descriptors_.row(idx));
            if (distance < bestDist)
            {
                bestDist = distance;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= TH_LOW)
        {
            MapPoint* mpOrg = keyframe->mappoints_[bestIdx];
            if (mpOrg)
            {
                if (!mpOrg->isBad())
                {
                    if (mpOrg->getObsCnt() > mp->getObsCnt())
                        mp->replaceMapPoint(mpOrg);
                    else
                        mpOrg->replaceMapPoint(mp);
                }
            }
            else
            {
                mp->addObservation(keyframe, bestIdx);
                keyframe->addMapPoint(mp, bestIdx);
            }
            
            cnt++;
        }
    }
    
    return cnt;
}

int Matcher::fuseByPose(KeyFrame* keyframe, Sophus::Sim3 &Scw, vector<MapPoint*> &loopMapPoints,
                        vector<MapPoint*> &replaceMapPoints, const float th)
{
    Camera* camera = keyframe->camera_;
    const float &fx = camera->fx_;
    const float &fy = camera->fy_;
    const float &cx = camera->cx_;
    const float &cy = camera->cy_;
    
    SE3 Tcw(Scw.rotation_matrix(), Scw.translation());
    Vector3d Ow = -Tcw.rotation_matrix().transpose() * Tcw.translation();
    
    set<MapPoint*> alreadyFound;
    for (int i = 0, N = keyframe->mappoints_.size(); i < N; i++)
    {
        MapPoint* mp = keyframe->mappoints_[i];
        if (!mp || mp->isBad())
            continue;
        alreadyFound.insert(mp);
    }
    
    int fused = 0;
    for (int i = 0, N = loopMapPoints.size(); i < N; i++)
    {
        
        MapPoint* mp = loopMapPoints[i];
        if (!mp || mp->isBad() || alreadyFound.count(mp))
            continue;
        
        Vector3d p_cam = Tcw * mp->getPose();
        const float z = static_cast<float>(p_cam[2]);
        if (z < 0)
            continue;
        
        const float invz = 1.0f / z;
        const float x = static_cast<float>(p_cam[0]) * invz;
        const float y = static_cast<float>(p_cam[1]) * invz;
        const float u = fx*x + cx;
        const float v = fy*y + cy;
        
        if (!keyframe->isInImg(u, v))
            continue;
        
        const float max_distance = mp->getMaxDistanceThreshold();
        const float min_distance = mp->getMinDistanceThreshold();
        Vector3d pline = mp->getPose() - Ow;
        const float distance = pline.norm();
        
        if (distance < min_distance || distance > max_distance)
            continue;
        
        Vector3d pNormal = mp->getNormalVector();
        if (pline.dot(pNormal) < 0.5*distance)
            continue;
        
        int level_predict = mp->predictScale(distance, keyframe);
        const float radius = th * keyframe->scaleFactors_[level_predict];
        
        const vector<int> indexs = keyframe->getFeaturesInArea(u, v, radius);
        
        if (indexs.empty())
            continue;
        
        Mat desp_mp = mp->getDescriptor();
        int bestDist = 256;
        int bestIdx = -1;
        
        for (int j = 0, N = indexs.size(); j < N; j++)
        {
            const int idx = indexs[j];
            
            const int level = keyframe->unKeypoints_[idx].octave;
            if (level < level_predict - 1 || level > level_predict)
                continue;
            
            Mat desp_kf = keyframe->descriptors_.row(idx);
            const int dist = computeDistance(desp_mp, desp_kf);
            
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if (bestDist <= TH_LOW)
        {
            MapPoint* mpKF = keyframe->mappoints_[bestIdx];
            if (mpKF)
            {
                if (!mpKF->isBad())
                    replaceMapPoints[i] = mpKF;
            }
            else
            {
                mp->addObservation(keyframe, bestIdx);
                keyframe->addMapPoint(mp, bestIdx);
            }
            fused++;
        }
    }
    
    return fused;
}

int Matcher::computeDistance(const Mat& desp1, const Mat& desp2)
{
    const int *p1 = desp1.ptr<int32_t>();
    const int *p2 = desp2.ptr<int32_t>();
    
    int dist = 0;
    
    for(int i=0; i<8; i++, p1++, p2++)
    {
        unsigned  int v = *p1 ^ *p2;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
   
void Matcher::computeThreeMax(vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;
    
    // compute top3
    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            ind3 = ind2;
            
            max2 = max1;
            ind2 = ind1;
            
            max1 =s;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            ind3 = ind2;
            
            max2 = s;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }
    
    // discard unsignificant index
    if (max2 < 0.1f*static_cast<float>(max1))
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f*static_cast<float>(max1))
    {
        ind3 = -1;
    }
}

bool Matcher::checkEpipolarConstrain(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                                     const Eigen::Matrix3d &F12, const KeyFrame *keyframe2)
{
    
    Vector3d p1(kp1.pt.x, kp1.pt.y, 1);
    Vector3d p2(kp2.pt.x, kp2.pt.y, 1);
    Vector3d l2 = (p1.transpose()*F12).transpose();
    
    const float numerator = l2.dot(p2);
    const float denominator = l2[0]*l2[0] + l2[1]*l2[1];
    
    if ( denominator == 0)
        return false;
    
    const float d_square = numerator*numerator / denominator;
    const float sigma = keyframe2->scaleFactors_[kp2.octave];
    
    return d_square < 3.84f*sigma*sigma;  // 95%
}
   
}
