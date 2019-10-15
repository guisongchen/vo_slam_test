#include "myslam/config.h"

namespace myslam
{

Config* Config::config_ = static_cast<Config*>(nullptr);
    
Config::~Config()
{
    if (file_.isOpened())
        file_.release();
}
 
void Config::setParameterFile(const std::string &filename)
{
    if (config_ == nullptr)
        config_ = new Config;
    config_->file_ = cv::FileStorage (filename.c_str(), cv::FileStorage::READ);
    if (!config_->file_.isOpened())
    {
        cerr << "parameter file " << filename << " does not exist" << endl;
        config_->file_.release();
        return;
    }
}

 
}
