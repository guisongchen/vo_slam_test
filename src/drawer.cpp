#include "myslam/drawer.h"
#include "myslam/config.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>

namespace myslam
{
    
Drawer::Drawer(Map* map_curr) 
    : map_curr_(map_curr),mapTrackedCnt_(0), voTrackedCnt_(0), finishFlag_(false), finishRequestFlag_(false)
{
    
    width_ = Config::get<int>("drawer_width");
    height_ = Config::get<int>("drawer_height");
    fu_ = Config::get<double>("drawer_fu");
    fv_ = Config::get<double>("drawer_fv");
    u0_ = Config::get<double>("drawer_u0");
    v0_ = Config::get<double>("drawer_v0");
    viewpointX_ = Config::get<double>("drawer_viewpointX");
    viewpointY_ = Config::get<double>("drawer_viewpointY");
    viewpointZ_ = Config::get<double>("drawer_viewpointZ");
    
    img_curr_ = Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0));
}

void Drawer::run()
{
    pangolin::CreateWindowAndBind("vo_ccc: Drawer", width_, height_);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    pangolin::CreatePanel("ViewMenu").SetBounds(pangolin::Attach::Pix(670), 1.0, 
                                                0.0, pangolin::Attach::Pix(175));
    
    pangolin::Var<bool> menuFollowCamera("ViewMenu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("ViewMenu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("ViewMenu.Show KeyFrames",true,true);
    
    pangolin::CreatePanel("InfoMenu").SetBounds(pangolin::Attach::Pix(470), pangolin::Attach::Pix(670),
                                                0.0, pangolin::Attach::Pix(175));
    
    pangolin::Var<double> menuQuatX("InfoMenu.Quat:x");
    pangolin::Var<double> menuQuatY("InfoMenu.Quat:y");
    pangolin::Var<double> menuQuatZ("InfoMenu.Quat:z");
    pangolin::Var<double> menuQuatW("InfoMenu.Quat:w");
    
    pangolin::Var<double> menuTranX("InfoMenu.Tran:x");
    pangolin::Var<double> menuTranY("InfoMenu.Tran:y");
    pangolin::Var<double> menuTranZ("InfoMenu.Tran:z");
    
    pangolin::CreatePanel("SaveMenu").SetBounds(0.0, pangolin::Attach::Pix(470),
                                                0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuSaveVocabulary("SaveMenu.Save Vocabulary",false,true);
    
    
    pangolin::OpenGlRenderState s_cam(
              pangolin::ProjectionMatrix(width_, height_, fu_, fv_, u0_, v0_, 0.1, 1000),
              pangolin::ModelViewLookAt(viewpointX_,viewpointY_,viewpointZ_, 0.0,0.0,0.0, 0.0,-1.0,0.0) );
    
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0/768.0)
            .SetHandler(new pangolin::Handler3D(s_cam));
            
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    
    cv::namedWindow("vo_ccc: Current Frame");
    
    finishFlag_ = false;
    
    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        getOpenGLCameraMatrix(Twc);
        
        if (menuFollowCamera)
            s_cam.Follow(Twc);
    
        d_cam.Activate(s_cam);

        Eigen::Vector4d q = Tcw_.unit_quaternion().coeffs();
        menuQuatX = q[0];
        menuQuatY = q[1];
        menuQuatZ = q[2];
        menuQuatW = q[3];
        
        Vector3d t = Tcw_.translation();
        menuTranX = t[0];
        menuTranY = t[1];
        menuTranZ = t[2];
    
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawCamera(Twc);
        
        if (menuShowKeyFrames)
            drawKeyFrames();
        
        if (menuShowPoints)
            drawMapPoints();
        
        pangolin::FinishFrame();
        
        Mat imgProcessed = drawCurrFrameImg();
        
        cv::imshow("vo_ccc: Current Frame", imgProcessed);
        cv::waitKey(1);
        
        if (checkFinishRequest())
            break;
    }
    
    if (menuSaveVocabulary)
        map_curr_->saveVocabularyFlag_ = true;
    
    setFinish();
}


void Drawer::setCurrPose(SE3 &Tcw)
{
    unique_lock<mutex> lock(mutexPose_);
    Tcw_ = Tcw;
}

// openGL: column store order -> Eigen: column store order
void Drawer::getOpenGLCameraMatrix (pangolin::OpenGlMatrix &M)
{
    SE3 Twc;
    {
        unique_lock<mutex> lock(mutexPose_);
        Twc = Tcw_.inverse();
    }
    
    M = Twc.matrix();
}

void Drawer::drawMapPoints()
{
    const vector<MapPoint*> allMapPoints = map_curr_->getAllMapPoints();
    const vector<MapPoint*> localMapPoints = map_curr_->getLocalMapPoints();
    
    set<MapPoint*> localPoints(localMapPoints.begin(), localMapPoints.end());
    if (allMapPoints.empty())
        return;
    
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    
    for (size_t i = 0, N = allMapPoints.size(); i < N; i++)
    {
        MapPoint* mp = allMapPoints[i];
        if (!mp || localPoints.count(mp))
            continue;
        
        Vector3d pos = mp->pos_;
        glVertex3f(pos[0], pos[1], pos[2]);
    }
    glEnd();
    
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    
    for (size_t i = 0, N = localMapPoints.size(); i < N; i++)
    {
        MapPoint* mp = localMapPoints[i];
        if (!mp)
            continue;
        
        Vector3d pos = mp->pos_;
        glVertex3f(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

void Drawer::drawCamera(pangolin::OpenGlMatrix &Twc)
{
    const float w = 0.08f;
    const float h = w*0.75;
    const float z = w*0.6;
    
    glPushMatrix();
    
    glMultMatrixd(Twc.m);

    glLineWidth(3);

    glColor3f(0.0f,1.0f,0.0f);

    glBegin(GL_LINES);
    
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    
    glEnd();

    glPopMatrix();
}

void Drawer::drawTextOnImg(Mat &img, Mat &textImg, int state)
{
    stringstream text;
    if (state == VisualOdometry::INITILIZING)
        text << " INITILIZING...";
    else if (state == VisualOdometry::OK)
        text << " VO RUNNING | ";
    else if (state == VisualOdometry::LOST)
        text << " RELOCALIZATION | ";
    
    unsigned long keyframes_cnt = map_curr_->getAllKeyFramesCnt();
    unsigned long mappoints_cnt = map_curr_->getAllMapPointsCnt();
    
    text << "KFs: " << keyframes_cnt << " | MPs: " << mappoints_cnt 
         << " | mapTracked:" << mapTrackedCnt_ << " | VOTracked: " << voTrackedCnt_;
         
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
    
    textImg = Mat(img.rows + textSize.height + 10, img.cols, img.type());
    
    img.copyTo( textImg.rowRange(0,img.rows).colRange(0, img.cols) );
    
    textImg.rowRange(img.rows, textImg.rows) = Mat::zeros(textSize.height + 10, img.cols, img.type());
    
    cv::putText(textImg, text.str(), cv::Point(5, textImg.rows-5),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8);
}

void Drawer::drawKeyFrames()
{
    const float w = 0.05f;
    const float h = w*0.75;
    const float z = w*0.6;
    
    const vector<KeyFrame*> allKFs = map_curr_->getAllKeyFrames();
    for (size_t i = 0, N = allKFs.size(); i < N; i++)
    {
        KeyFrame* kf = allKFs[i];
        if (kf->isBad())
            continue;
        
        SE3 Twc = kf->Tcw_.inverse();
        
        glPushMatrix();
        pangolin::OpenGlMatrix M(Twc.matrix());
        
        glMultMatrixd(M.m);
        
        glLineWidth(3);
        glColor3f(0.0f,0.0f,1.0f);

        glBegin(GL_LINES);
        
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        
        glEnd();

        glPopMatrix();
    }
    
    glLineWidth(3);
    glBegin(GL_LINES);
    
    for (size_t i = 0, N = allKFs.size(); i < N; i++)
    {
        KeyFrame* kf = allKFs[i];
        if (kf->isBad())
            continue;
        
        const vector<KeyFrame*> covKFs = kf->getCovisiblesByWeight(100);
        Vector3d Ow = kf->getCamCenter();
        
        // green for covisble
        if (!covKFs.empty())
        {
            for (auto it = covKFs.begin(); it != covKFs.end(); it++)
            {
                if ((*it)->id_ == kf->id_)
                    continue;
                Vector3d Owc = (*it)->getCamCenter();
                
                glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
                glVertex3f(Ow[0], Ow[1], Ow[2]);
                glVertex3f(Owc[0], Owc[1], Owc[2]);
            }
        }
        
        // red for spanning tree
        KeyFrame* parentKF = kf->parent_;
        if (parentKF)
        {
            Vector3d Owp = parentKF->getCamCenter();
            
            glColor4f(1.0f, 0.0f, 0.0f, 0.6f);
            glVertex3f(Ow[0], Ow[1], Ow[2]);
            glVertex3f(Owp[0], Owp[1], Owp[2]);
        }
        
        // bule for loop edge
        set<KeyFrame*> loopKFs = kf->loopEdges_;
        if (!loopKFs.empty())
        {
            for (auto it = loopKFs.begin(); it != loopKFs.end(); it++)
            {
                if ((*it)->id_ == kf->id_)
                    continue;
                
                Vector3d Owl = (*it)->getCamCenter();
                
                glColor4f(0.0f, 0.0f, 1.0f, 0.6f);
                glVertex3f(Ow[0], Ow[1], Ow[2]);
                glVertex3f(Owl[0], Owl[1], Owl[2]);
            }
        }
    }
    
    glEnd();
    
}

Mat Drawer::drawCurrFrameImg()
{
    Mat img;
    vector<cv::KeyPoint> keypoints_curr;
    vector<bool>         inMapFlag;
    vector<bool>         inVOFlag;
    int state;
    
    {
        unique_lock<mutex> lock(mutexFrame_);
        
        img_curr_.copyTo(img);
        keypoints_curr = keypoints_curr_;
        inMapFlag = inMapFlag_;
        inVOFlag = inVOFlag_;
        state = state_;
    }
    
    if (img.channels() < 3)
        cvtColor(img, img, CV_GRAY2RGB);
    
    mapTrackedCnt_ = 0;
    voTrackedCnt_ = 0;
    
    const float r = 5;
    for (size_t i = 0; i < N_; i++)
    {
        if (inMapFlag[i] || inVOFlag[i])
        {
            cv::Point2f pt1, pt2;
            cv::KeyPoint kpt = keypoints_curr[i];
            
            pt1.x = kpt.pt.x - r;
            pt1.y = kpt.pt.y - r;
            pt2.x = kpt.pt.x + r;
            pt2.y = kpt.pt.y + r;
            
            if (inMapFlag[i])
            {
                cv::rectangle(img, pt1, pt2, cv::Scalar(0,255,0));
                cv::circle(img, kpt.pt, 2, cv::Scalar(0,255,0), -1);
                mapTrackedCnt_++;
            }
            
            else
            {
                cv::rectangle(img, pt1, pt2, cv::Scalar(255,0,0));
                cv::circle(img, kpt.pt, 2, cv::Scalar(255,0,0), -1);
                voTrackedCnt_++;
            }
            
        }
    }
    
    Mat textImg;
    drawTextOnImg(img, textImg, state);
    
    return textImg;
    
}

void Drawer::updateCurrFrame(VisualOdometry* vo_curr)
{
    {
        unique_lock<mutex> lock(mutexFrame_);
        vo_curr->grayImg_.copyTo(img_curr_);
        keypoints_curr_ = vo_curr->frame_curr_->keypoints_;
    }
    
    state_ = vo_curr->lastState_;
    N_ = keypoints_curr_.size();
    
    inMapFlag_ = vector<bool>(N_, false);
    inVOFlag_ = vector<bool>(N_, false);
    
    Frame* frame_curr = vo_curr->frame_curr_;
    if (state_ == VisualOdometry::OK)
    {
        for (size_t i = 0; i < N_; i++)
        {
            MapPoint* mp = frame_curr->mappoints_[i];
            
            if (mp && !frame_curr->outliers_[i])
            {
                if (mp->observe_cnt_ > 0)
                    inMapFlag_[i] = true;
                else
                    inVOFlag_[i] = true;
            }
        }
    }
}

void Drawer::requestFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    finishRequestFlag_ = true;
}

void Drawer::setFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    finishFlag_ = true;
}

bool Drawer::checkFinishRequest()
{
    unique_lock<mutex> lock(mutexFinish_);
    return finishRequestFlag_;
}

bool Drawer::checkFinish()
{
    unique_lock<mutex> lock(mutexFinish_);
    return finishFlag_;
}

}
