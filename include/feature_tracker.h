#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H

#include "camera_model/camera.h"
#include "parameter.h"

namespace myslam
{
    
bool inBorder(const cv::Point2f &pt);

void cullingOutBorders_(std::vector<cv::Point2f> &v, std::vector<uchar> status);
void cullingOutBorders_(std::vector<int> &v, std::vector<uchar> status);
    
class VisualTracker
{
public:
    VisualTracker();
    
    void trackImageFeature(const Mat &img, double currTime);
    void setMask();
    void addPoints();
    bool updateID(unsigned int i);
    void readCameraConfig(const std::string &calibFile);
    void cullingByF();
    void undistortedPtsAndComputeVel();
    
   
    static int  factoryId_;
    CameraPtr   camera_;
    
    Mat         mask_;
    Mat         lastImg_, currImg_;
    double      lastTime_, currTime_;
    
    std::vector<cv::Point2f>    addedPoints_;
    std::vector<cv::Point2f>    lastPoints_, currPoints_;
    std::vector<cv::Point2f>    lastUndistortPoints_, currUnDistortPoints_;
    std::vector<cv::Point2f>    pointsVelocity_;
    std::vector<int>            trackedFeatureIds_;
    std::vector<int>            trackedCnts_;
    std::map<int, cv::Point2f>  currUndistortPointsMap_;
    std::map<int, cv::Point2f>  lastUndistortPointsMap_;
};

}

#endif
