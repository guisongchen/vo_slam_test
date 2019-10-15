#include "myslam/config.h"
#include "myslam/visualOdometry.h"
#include "myslam/map.h"
#include "myslam/localMapping.h"
#include "myslam/loopClosing.h"
#include "myslam/drawer.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <DBoW3/DBoW3.h>
#include <thread>
#include <chrono>

int main (int argc, char** argv)
{
    if (argc != 2)
    {
        cerr << "ERROR usage:vo_run path of parameter file" << endl;
        return 1;
    }
    
    myslam::Config::setParameterFile(argv[1]);
    
    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    int looptime = myslam::Config::get<int>("data_num");
//     string out_path = myslam::Config::get<string>("out_path");
    
    ifstream fin(dataset_dir+"/associate.txt");
//     ofstream fout(out_path);
 
    if (fin.fail())
    {
        cerr << "ERROR, can't find associate file in " << dataset_dir << endl;
        return 1;
    }
    
    vector<string> rgb_files, depth_files;
    vector<string> rgb_times, depth_times;
    
    // NOTE: use looptime to limit the number of images to be processed, change it!! 
    int imgCnt = 0;
    for (int i = 0; i < looptime; i++)
    {
        if (fin.eof())
        {
            cout << "file end before loop finish." << endl;
            break;
        }
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back( rgb_time );
        rgb_files.push_back( dataset_dir + rgb_file );
        depth_times.push_back( depth_time );
        depth_files.push_back(dataset_dir + depth_file);

        imgCnt++;
    }
    fin.close();
    
    cout << "total image number: " << imgCnt << endl;
    
    myslam::Map* myMap(new myslam::Map);
    myslam::Drawer* drawer(new myslam::Drawer(myMap));
    myslam::Camera* camera(new myslam::Camera);
    myslam::VisualOdometry* vo(new myslam::VisualOdometry(myMap, camera, drawer));
    myslam::LocalMapping* localMapper(new myslam::LocalMapping(myMap));
    myslam::LoopClosing* loopCloser(new myslam::LoopClosing(myMap));
    
    
    vo->localMapper_ = localMapper;
    localMapper->vo_ = vo;
    thread* localMapping = new thread(&myslam::LocalMapping::run, localMapper);
    
    camera->printCameraInfo();
    
    cout << "load vocabulary....." << endl;
    string voc_path = myslam::Config::get<string>("vocabulary_in");
    ifstream ifile(voc_path); 
    
    if (!ifile)
    {
        cout << "vocabulary file not exist." << endl;
        return 1;
    }

    DBoW3::Vocabulary* vocab = new DBoW3::Vocabulary(voc_path);
    
    vo->voc_ = vocab;
    vo->map_->voc_ = vocab;
    vo->map_->invertIdxs_.resize(vocab->size());
    
    cout << "load vocabulary successfully.\npath:"<< voc_path << "; size:" << vocab->size() << endl;
    
    loopCloser->setVocabulary(vocab);
    thread* loopclosing = new thread(&myslam::LoopClosing::run, loopCloser);
    
    loopCloser->setLocalMapper(localMapper);
    localMapper->loopCloser_ = loopCloser;
    
    thread* drawing = new thread(&myslam::Drawer::run, drawer);
    
    vector<double> timeCosts;
    
    int lostCnt = 0;
    for (int i = 0; i < imgCnt; i++)
    {
//         cout << "***************** loop " << i << " *****************" << endl;
        
        Mat color = cv::imread(rgb_files[i], 1);
        Mat depth = cv::imread(depth_files[i], -1);
        if (!color.data || !depth.data)
        {
            cout << "no more image info." << endl;
            break;
        }
        
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        
        vo->run(color, depth, rgb_times[i]);
        
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        
        double trackCost = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
        
        if (vo->num_lost_ > 0)
        {
            lostCnt++;
            continue;
        }
        
        timeCosts.push_back(trackCost);
        
//         // use this print pose before optimized 
//         SE3 Twc = vo->frame_curr_->Tcw_.inverse();
//         fout << rgb_times[i] << ' ' << Twc.translation().transpose() << ' ' 
//              << Twc.unit_quaternion().coeffs().transpose() << endl;
    }
    
//     fout.close();
    
    localMapper->requestFinish();
    loopCloser->requestFinish();
    drawer->requestFinish();
    
    while (1)
    {
        if (localMapper->checkFinish() && drawer->checkFinish() && loopCloser->checkFinish())
            break;
    }
    
    sort(timeCosts.begin(), timeCosts.end());
    double totalTime = 0.0;
    for (auto t:timeCosts)
        totalTime += t;
    
    int trackCnt = imgCnt - lostCnt;
    cout << "total tracked number: " << trackCnt << "; total lost times: " << lostCnt << endl;
    cout << "median tracking time: " << timeCosts[trackCnt/2] << endl;
    cout << "mean tracking time: " << totalTime/trackCnt << endl;
    
    cout << "start saving keyframe trajectory..." << endl;
    
    string keyframe_path = myslam::Config::get<string>("keyframe_path");
    ofstream fKeyframe(keyframe_path);
    
    vector<myslam::KeyFrame*> keyframes = myMap->getAllKeyFrames();
    sort(keyframes.begin(), keyframes.end(), 
         [] (const myslam::KeyFrame* kf1, const myslam::KeyFrame* kf2) {return kf1->id_ < kf2->id_;});
    
    for (size_t i = 0, N = keyframes.size(); i < N; i++)
    {
        myslam::KeyFrame* kf = keyframes[i];
        
        if (kf->isBad())
            continue;
        
        SE3 Twc =(kf->getPose()).inverse();
        
        fKeyframe << kf->timeStamp_ << " " << Twc.translation().transpose() << " " 
                  << Twc.unit_quaternion().coeffs().transpose() << endl;
    }

    fKeyframe.close();
    cout << "keyframe trajectory saved !!!" << endl;
    
    
    cout << "start saving camera trajectory..." << endl;
    string camera_path = myslam::Config::get<string>("camera_path");
    ofstream fCamera(camera_path);
    
    auto itTcr = vo->TcrDB_.begin();
    auto itState = vo->stateDB_.begin();
    auto itTime = vo->timestampDB_.begin();
    
    for (auto itRef = vo->trackRefKeyFrameDB_.begin(), itRefEnd = vo->trackRefKeyFrameDB_.end();
         itRef != itRefEnd; itRef++, itTcr++, itState++, itTime++)
    {
        if (!(*itState))
            continue;
        
        myslam::KeyFrame* keyframe_ref = *itRef;
        
        SE3 Twc;
        SE3 &Tcr = *itTcr;
        
        if (!keyframe_ref->isBad())
        {
            SE3 Trw = keyframe_ref->getPose();
            SE3 Tcw = Tcr * Trw;
            Twc = Tcw.inverse();
        }
        else
        {
            SE3 Trp = SE3();
            
            while (keyframe_ref->isBad())
            {
                Trp = Trp * keyframe_ref->Tcp_;
                keyframe_ref = keyframe_ref->getParent();
            }
            
            SE3 Tpw = keyframe_ref->getPose();
            SE3 Tcw = Tcr * Trp * Tpw;
            Twc = Tcw.inverse();
        }
        
        fCamera << *itTime << " " << Twc.translation().transpose() << " " 
             << Twc.unit_quaternion().coeffs().transpose() << endl;
    }
    
    fCamera.close();
    cout << "camera trajectory saved !!!" << endl;
    
    vo->map_->createVocabulary();
    
    return 0;
}
