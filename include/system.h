#ifndef SYSTEM_H
#define SYSTEM_H

#include "feature_tracker.h"
#include "estimator.h"

#include <unistd.h>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <pangolin/pangolin.h>

namespace myslam
{
    
struct IMU_MSG
{
    double      header_;
    Vector3d    linearAcc_;
    Vector3d    angularVel_;
};
typedef std::shared_ptr<const IMU_MSG> ImuConstPtr;

struct IMG_MSG {
    double  header_;
    std::vector<Vector3d>   p3dsNormalized_;
    std::vector<int>        pointsId_;
    std::vector<float>      pixelsU_;
    std::vector<float>      pixelsV_;
    std::vector<float>      velocitysX_;
    std::vector<float>      velocitysY_;
};
typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;
    
class System
{
public:
    System(std::string configFile);
    ~System();
    
    void pubImgData(double stampSec, Mat &img);
    void pubImuData(double stampSec, const Vector3d &gyrInfo, const Vector3d &accInfo);
    void processBackEnd();
    
    void Draw();
    void drawCamera(pangolin::OpenGlMatrix& Twc);
    
    bool ShutDown() const { return shutDownFlag_; }
    
private:
    
    VisualTracker   tracker_;
    Estimator       estimator_;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
    
    bool            startBackEndFlag_;
    std::ofstream   poseFile_;
    
    double          lastImuTime_;
    std::mutex      mutexBuf_;
    std::condition_variable condVar_;
    std::queue<ImuConstPtr> imuBuf_;
    
    bool            initFeatureFlag_;
    bool            firstImgFlag_;
    double          firstImgTime_;
    double          lastImgTime_;
    int             imgPubCnt_;
    bool            initImgPubFlag_;
    std::queue<ImgConstPtr> imgBuf_;
    
    int             sumOfWait_;
    std::mutex      mutexEstimate_;
    
    double          lastProcessTime_;
    std::vector<Eigen::Vector3d> cameraPaths_;
    pangolin::OpenGlMatrix Twc_;

    cv::Mat         show_img;
    bool            shutDownFlag_;
   
};
    
}

#endif
