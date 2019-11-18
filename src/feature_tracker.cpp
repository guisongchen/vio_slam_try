#include "feature_tracker.h"
#include "camera_model/camera_factory.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace myslam
{

int VisualTracker::factoryId_ = 0;
    
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x<(COL-BORDER_SIZE) && BORDER_SIZE <= img_y && img_y<(ROW - BORDER_SIZE);
}

void cullingOutBorders_(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void cullingOutBorders_(std::vector<int> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
    
VisualTracker::VisualTracker() {}

void VisualTracker::readCameraConfig(const std::string &calibFile)
{
    cout << "load camera parameter from " << calibFile << endl;
    camera_ = CameraFactory::generateCameraInstance()->generateCameraFromYamlFile(calibFile);
}

void VisualTracker::trackImageFeature(const cv::Mat &img, double currTime)
{
    cv::Mat imgEqualized;
    currTime_ = currTime;

    // histogram equalization: enforce image contrast, make image histogram equaly distributed
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, imgEqualized);
    }
    else
        imgEqualized = img;

    if (lastImg_.empty()) // should only for first image
        lastImg_ = currImg_ = imgEqualized;
    else
        currImg_ = imgEqualized;

    currPoints_.clear();

    // first and second image will jump this part since currPoints_ is empty
    if (lastPoints_.size() > 0)
    {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(lastImg_, currImg_, lastPoints_, currPoints_, status, err, cv::Size(21, 21), 3);

        // culling points out of border
        const int pointNum = static_cast<int>(currPoints_.size());
        for (int i = 0; i < pointNum; i++)
            if (status[i] && !inBorder(currPoints_[i]))
                status[i] = 0;
        
        cullingOutBorders_(lastPoints_, status);
        cullingOutBorders_(currPoints_, status);
        cullingOutBorders_(trackedFeatureIds_, status);
        cullingOutBorders_(currUnDistortPoints_, status);
        cullingOutBorders_(trackedCnts_, status);
    }

    for (auto &trackedCnt : trackedCnts_)
        trackedCnt++;

    if (PUB_THIS_FRAME)
    {
        // second image will do nothing except init mask interest region as full size
        cullingByF();
        setMask();

        // second image will add all MAX_CNT points. following images will depend on tracking result
        int addedPointsCnt = MAX_CNT - static_cast<int>(currPoints_.size());
        if (addedPointsCnt > 0)
        {
            if (mask_.empty())
                cout << "mask is empty " << endl;
            if (mask_.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask_.size() != currImg_.size())
                cout << "wrong size " << endl;
            
            // detect corners of input image
            cv::goodFeaturesToTrack(currImg_, addedPoints_, addedPointsCnt, 0.01, MIN_DIST, mask_);
        }
        else
            addedPoints_.clear();

        // init currPoints_ if second image, add currPoints_ if following images 
        addPoints();
    }
    
    // if second image, last info will be init
    lastImg_ = currImg_;
    lastPoints_ = currPoints_;
    lastUndistortPoints_ = currUnDistortPoints_;
    
    // start getting currUnDistortPoints_ and prevUndistortPointsMap_ and velocity from second image
    undistortedPtsAndComputeVel();
    
    lastTime_ = currTime_;
}

void VisualTracker::cullingByF()
{
    if (currPoints_.size() >= 8)
    {
        
        std::vector<cv::Point2f> undistortCurrPoints(lastPoints_.size()), undistortForwPoints(currPoints_.size());
        for (unsigned int i = 0, N = lastPoints_.size(); i < N; i++)
        {
            // using virtual camera model here, FOCAL_LENGTH is hardcorded, cx/cy set as half of image size
            
            Eigen::Vector3d tmpPoint;
            camera_->liftProjective(Vector2d(lastPoints_[i].x, lastPoints_[i].y), tmpPoint);
            tmpPoint.x() = FOCAL_LENGTH * tmpPoint.x() / tmpPoint.z() + COL / 2.0;
            tmpPoint.y() = FOCAL_LENGTH * tmpPoint.y() / tmpPoint.z() + ROW / 2.0;
            undistortCurrPoints[i] = cv::Point2f(tmpPoint.x(), tmpPoint.y());

            camera_->liftProjective(Eigen::Vector2d(currPoints_[i].x, currPoints_[i].y), tmpPoint);
            tmpPoint.x() = FOCAL_LENGTH * tmpPoint.x() / tmpPoint.z() + COL / 2.0;
            tmpPoint.y() = FOCAL_LENGTH * tmpPoint.y() / tmpPoint.z() + ROW / 2.0;
            undistortForwPoints[i] = cv::Point2f(tmpPoint.x(), tmpPoint.y());
        }

        std::vector<uchar> status;
        cv::findFundamentalMat(undistortCurrPoints, undistortForwPoints, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        
        cullingOutBorders_(lastPoints_, status);
        cullingOutBorders_(currPoints_, status);
        cullingOutBorders_(trackedFeatureIds_, status);
        cullingOutBorders_(currUnDistortPoints_, status);
        cullingOutBorders_(trackedCnts_, status);
    }
}

void VisualTracker::setMask()
{
    // in fact, if we don't use fisheye camera, all the region is interest zone, so the value is 255
    mask_ = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    std::vector< std::pair< int, std::pair<cv::Point2f, int> > > tracked_cnt_pts_id;

    for (unsigned int i = 0; i < currPoints_.size(); i++)
        tracked_cnt_pts_id.push_back(std::make_pair(trackedCnts_[i], std::make_pair(currPoints_[i], trackedFeatureIds_[i])));

    sort(tracked_cnt_pts_id.begin(), tracked_cnt_pts_id.end(), 
         [](const std::pair<int, std::pair<cv::Point2f, int> > &a,
            const std::pair<int, std::pair<cv::Point2f, int> > &b)
         { return a.first > b.first; } );

    currPoints_.clear();
    trackedFeatureIds_.clear();
    trackedCnts_.clear();

    for (auto &it : tracked_cnt_pts_id)
    {
        if (mask_.at<uchar>(it.second.first) == 255)
        {
            currPoints_.push_back(it.second.first);
            trackedFeatureIds_.push_back(it.second.second);
            trackedCnts_.push_back(it.first);
            cv::circle(mask_, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void VisualTracker::addPoints()
{
    for (auto &point : addedPoints_)
    {
        currPoints_.push_back(point);
        trackedFeatureIds_.push_back(-1);
        trackedCnts_.push_back(1);
    }
}

void VisualTracker::undistortedPtsAndComputeVel()
{
    currUnDistortPoints_.clear();
    currUndistortPointsMap_.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    
    for (unsigned int i = 0; i < lastPoints_.size(); i++)
    {
        Eigen::Vector2d p2d(lastPoints_[i].x, lastPoints_[i].y);
        Eigen::Vector3d p3d;
        
        // undistorted operation happens here
        camera_->liftProjective(p2d, p3d);
        
        currUnDistortPoints_.push_back(cv::Point2f(p3d.x() / p3d.z(), p3d.y() / p3d.z()));
        currUndistortPointsMap_.insert(std::make_pair(trackedFeatureIds_[i],
                                       cv::Point2f(p3d.x() / p3d.z(), p3d.y() / p3d.z())));
        
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    
    // caculate points velocity
    if (!lastUndistortPointsMap_.empty())
    {
        double dt = currTime_ -lastTime_;
        pointsVelocity_.clear();
        
        for (unsigned int i = 0; i < currUnDistortPoints_.size(); i++)
        {
            // choosen pts tracked by many images, not including fresh added after culling
            if (trackedFeatureIds_[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = lastUndistortPointsMap_.find(trackedFeatureIds_[i]);
                if (it != lastUndistortPointsMap_.end())
                {
                    double v_x = (currUnDistortPoints_[i].x - it->second.x) / dt;
                    double v_y = (currUnDistortPoints_[i].y - it->second.y) / dt;
                    pointsVelocity_.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pointsVelocity_.push_back(cv::Point2f(0, 0));
            }
            else
                pointsVelocity_.push_back(cv::Point2f(0, 0));
        }
    }
    else
    {
        for (unsigned int i = 0; i < lastPoints_.size(); i++)
            pointsVelocity_.push_back(cv::Point2f(0, 0));
    }
    
    lastUndistortPointsMap_ = currUndistortPointsMap_;
}

bool VisualTracker::updateID(unsigned int i)
{
    if (i < trackedFeatureIds_.size())
    {
        if (trackedFeatureIds_[i] == -1)
            trackedFeatureIds_[i] = factoryId_++;
        return true;
    }
    else
        return false;
}
    
}
