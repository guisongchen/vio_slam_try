#include "system.h"
#include <set>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.h>

#include <Eigen/Geometry>

namespace myslam
{
    
System::System(std::string configPath )
    : startBackEndFlag_(true), lastImuTime_(0.0), initFeatureFlag_(false), firstImgFlag_(true),
      firstImgTime_(0.0), lastImgTime_(0.0), imgPubCnt_(1), initImgPubFlag_(true),
      sumOfWait_(0), lastProcessTime_(-1), shutDownFlag_(false)
{
    std::string configFile = configPath + "euroc_config.yaml";

    cout << "configFile: " << configFile << endl;
    readParameters(configFile);

    tracker_.readCameraConfig(configFile);
    estimator_.setParameter();
    
    poseFile_.open("../result/output_pose.txt", std::fstream::out);
    if(!poseFile_.is_open())
        std::cerr << "pose file is not open" << endl;
}

System::~System()
{
    startBackEndFlag_ = false;
    
    pangolin::QuitAll();
    
    mutexBuf_.lock();
    
    while (!imgBuf_.empty())
        imgBuf_.pop();
    
    while (!imuBuf_.empty())
        imuBuf_.pop();
    
    mutexBuf_.unlock();

    mutexEstimate_.lock();
    estimator_.clearState();
    mutexEstimate_.unlock();

    poseFile_.close();
}

// thread: visual-inertial odometry
void System::processBackEnd()
{
    cout << "ProcessBackEnd start" << endl;
    
    while (startBackEndFlag_)
    {
        if (shutDownFlag_)
            break;
        
        std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr> > measurements;
        
        std::unique_lock<std::mutex> lock(mutexBuf_);
        condVar_.wait(lock, [&] { return shutDownFlag_ || (measurements = getMeasurements()).size() != 0; });
        lock.unlock();
        
        mutexEstimate_.lock();
        for (auto &measurement : measurements)
        {
            auto imageMsg = measurement.second;
            
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            double img_t = imageMsg->header_ + estimator_.td_;
            double ba[]{0.0, 0.0, 0.0};
            double bg[]{0.0, 0.0, 0.0};
            
            for (auto &imuMsg : measurement.first)
            {
                double imu_t = imuMsg->header_;
                
                if (imu_t <= img_t)
                {
                    if (lastProcessTime_ < 0)
                        lastProcessTime_ = imu_t;
                    
                    double dt = imu_t - lastProcessTime_;
                    lastProcessTime_ = imu_t;
                    
                    assert(dt >= 0);
                    
                    dx = imuMsg->linearAcc_.x() - ba[0];
                    dy = imuMsg->linearAcc_.y() - ba[1];
                    dz = imuMsg->linearAcc_.z() - ba[2];
                    
                    rx = imuMsg->angularVel_.x() - bg[0];
                    ry = imuMsg->angularVel_.y() - bg[1];
                    rz = imuMsg->angularVel_.z() - bg[2];
                    
                    estimator_.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else // t > img_t, this may happend when last imu in queue is slaightly large than img_t
                {
                    // interpolation between t and img_t
                    double dt_1 = img_t - lastProcessTime_;
                    double dt_2 = imu_t - img_t;
                    lastProcessTime_ = img_t;
                    
                    assert(dt_1 >= 0 && dt_2 >= 0 && dt_1 + dt_2 > 0);
                    
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    
                    dx = w1 * dx + w2 * imuMsg->linearAcc_.x() - ba[0];
                    dy = w1 * dy + w2 * imuMsg->linearAcc_.y() - ba[1];
                    dz = w1 * dz + w2 * imuMsg->linearAcc_.z() - ba[2];
                    
                    rx = w1 * rx + w2 * imuMsg->angularVel_.x() - bg[0];
                    ry = w1 * ry + w2 * imuMsg->angularVel_.y() - bg[1];
                    rz = w1 * rz + w2 * imuMsg->angularVel_.z() - bg[2];
                    
                    estimator_.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }

            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > image;
            const int pointNum = static_cast<int>(imageMsg->p3dsNormalized_.size());
            
            for (unsigned int i = 0; i < pointNum; i++) 
            {
                int v = imageMsg->pointsId_[i] + 0.5;
                
                // if more than one camera, distinguish them
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                
                // points belongs to normalized plane, which z == 1
                double x = imageMsg->p3dsNormalized_[i].x();
                double y = imageMsg->p3dsNormalized_[i].y();
                double z = imageMsg->p3dsNormalized_[i].z();
                
                // pixel and velocity
                double p_u = imageMsg->pixelsU_[i];
                double p_v = imageMsg->pixelsV_[i];
                double velocity_x = imageMsg->velocitysX_[i];
                double velocity_y = imageMsg->velocitysY_[i];
                
                // make sure points on normalized plane
                assert(z == 1);
                
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            estimator_.processImage(image, imageMsg->header_);
            
            if (estimator_.solverFlag_ == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator_.Rs[WINDOW_SIZE]);
                p_wi = estimator_.Ps[WINDOW_SIZE];
                
                Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
                T.rotate(estimator_.Rs[WINDOW_SIZE]);
                T.pretranslate(p_wi);
                Twc_ = T.matrix();
                
                cameraPaths_.push_back(p_wi);
                
                double dStamp = estimator_.headersInWindow_[WINDOW_SIZE];
                
                poseFile_ << std::fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " " 
                          << q_wi.w() << " " << q_wi.x() << " " << q_wi.y() << " " << q_wi.z() << endl;
            }
        }
        mutexEstimate_.unlock();
    }
    
    cout << "ProcessBackEnd shutdown" << endl;
}

std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr> > System::getMeasurements()
{
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr> > measurements;

    while (true)
    {
        if (imuBuf_.empty() || imgBuf_.empty())
        {
//             std::cerr << "imuBuf || imgBuf empty!!" << endl;
            return measurements;
        }
        
        // imu clock is accurate, real img(imu) clock = read img clock + td;
        // if we get img data, but imu date was left behind, should wait for imu
        // aka: last imu(newest time) <= first img(oldest time)
        if (!(imuBuf_.back()->header_ > imgBuf_.front()->header_ + estimator_.td_))
        {
            std::cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                      << sumOfWait_ << endl;
            sumOfWait_++;
            return measurements;
        }

        // img doesn't have corresponding imu data, throw img
        // aka: first imu(oldest time) >= first img(oldest time)
        if (!(imuBuf_.front()->header_ < imgBuf_.front()->header_ + estimator_.td_))
        {
            std::cerr << "throw img, only should happen at the beginning" << endl;
            imgBuf_.pop();
            continue;
        }
        
        ImgConstPtr imgMsg = imgBuf_.front();
        imgBuf_.pop();

        std::vector<ImuConstPtr> IMUs;
        
        // aligen imu and img timestamp
        // collect imu data until: imu timestamp > img timestamp
        while (imuBuf_.front()->header_ <= imgMsg->header_ + estimator_.td_)
        {
            // emplace_back: performance up. construct in place, neither copy or move objects like push_back does.
            IMUs.emplace_back(imuBuf_.front());
            imuBuf_.pop();
        }
        // cout << "1 getMeasurements IMUs size: " << IMUs.size() << endl;
        
        if (IMUs.empty())
            std::cerr << "no imu between two image" << endl;
        
        // get one img data and many imu data(which timestamp <= img timestamp)
        measurements.emplace_back(IMUs, imgMsg);
    }
    return measurements;
}

void System::pubImuData(double stampSec, const Vector3d &gyr, const Vector3d &acc)
{
    std::shared_ptr<IMU_MSG> imuMsg(new IMU_MSG());
    
    imuMsg->header_ = stampSec;
    imuMsg->linearAcc_ = acc;
    imuMsg->angularVel_ = gyr;

    if (stampSec <= lastImuTime_)
    {
        std::cerr << "imu message disorder!" << endl;
        return;
    }
    
    lastImuTime_ = stampSec;
    
    mutexBuf_.lock();
    imuBuf_.push(imuMsg);
    mutexBuf_.unlock();
    
    condVar_.notify_one();
}

void System::pubImgData (double stampSec, Mat &img)
{
    // first frame tracked in window in fact is second frame in image squence
    // because first image in squence do not have optical flow speed
    if (firstImgFlag_)
    {
//         cout << "PubImageData first_image_flag" << endl;
        firstImgFlag_ = false;
        firstImgTime_ = stampSec;
        lastImgTime_ = stampSec;
    }
    
    // detect unstable camera stream(missing 10 frames or disorder)
    if (stampSec - lastImgTime_ > 1.0 || stampSec < lastImgTime_)
    {
        std::cerr << "PubImageData image discontinue! reset the feature tracker!" << endl;
        firstImgFlag_ = true;
        lastImgTime_ = 0.0;
        imgPubCnt_ = 1;
        return;
    }
    
    lastImgTime_ = stampSec;
    
    // frequency control, less than 10fps
    // if first image, will NOT PUB, since currFREQ is big value: stampSec == firstImgTime_
    const int currFREQ = round((1.0*imgPubCnt_) / (stampSec - firstImgTime_));
    
    if (currFREQ <= FREQ)
    {
        PUB_THIS_FRAME = true;
        
        // reset the frequency control, why?
        if (abs(currFREQ - FREQ) < 0.01 * FREQ)
        {
            firstImgTime_ = stampSec;
            imgPubCnt_ = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // for first image only envalue lastImg_ = currImg_ = imgProcessed;
    // start from second image, begin tracking
    tracker_.trackImageFeature(img, stampSec);
    
    for (unsigned int i = 0;; i++)
    {
        // i >= trackedIds_.size, break
        if (!tracker_.updateID(i))
            break;
    }
    
    if (PUB_THIS_FRAME)
    {
        imgPubCnt_++;
        std::shared_ptr<IMG_MSG> featurePoints(new IMG_MSG());
        featurePoints->header_ = stampSec;

        auto &un_pts = tracker_.currUnDistortPoints_;
        auto &cur_pts = tracker_.currPoints_;
        auto &ids = tracker_.trackedFeatureIds_;
        auto &pts_velocity = tracker_.pointsVelocity_;
        
        const int pointNum = static_cast<int>(ids.size());
        for (unsigned int j = 0; j < pointNum; j++)
        {
            // collect points which at last observed by 2 frames 
            if (tracker_.trackedCnts_[j] > 1)
            {
                int p_id = ids[j];
                
                double x = un_pts[j].x;
                double y = un_pts[j].y;
                double z = 1;
                
                featurePoints->p3dsNormalized_.push_back(Vector3d(x, y, z));
                featurePoints->pointsId_.push_back(p_id);
                featurePoints->pixelsU_.push_back(cur_pts[j].x);
                featurePoints->pixelsV_.push_back(cur_pts[j].y);
                featurePoints->velocitysX_.push_back(pts_velocity[j].x);
                featurePoints->velocitysY_.push_back(pts_velocity[j].y);
            }
        }
        
        // skip the first image; since no optical speed on frist image
        if (initImgPubFlag_)
        {
//             cout << "PubImage init_pub skip the first image!" << endl;
            initImgPubFlag_ = false;
        }
        else
        {
            mutexBuf_.lock();
            imgBuf_.push(featurePoints);
            mutexBuf_.unlock();
            
            condVar_.notify_one();
        }
    }

	cv::cvtColor(img, show_img, CV_GRAY2RGB);
	if (SHOW_TRACK)
	{
		for (unsigned int j = 0; j < tracker_.lastPoints_.size(); j++)
        {
			double len = std::min(1.0, 1.0 * tracker_.trackedCnts_[j] / WINDOW_SIZE);
			cv::circle(show_img, tracker_.lastPoints_[j], 2, cv::Scalar(255 * (1-len), 0, 255*len), 2);
		}
	}
    
}

void System::Draw() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 0, 0, 0, 1.0, 0.0, 0.0)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
                           .SetBounds(0, 1.0, 0, 1.0, -1024.0/768.0)
                           .SetHandler(new pangolin::Handler3D(s_cam));
            
    pangolin::View& cam_img = pangolin::Display("rgb")
              .SetBounds(0.3, 0, 0, 0.3, -1024.0/768.0)
              .SetLock(pangolin::LockLeft, pangolin::LockBottom);
              
    pangolin::GlTexture imgTexture(752, 480, GL_RGB, GL_UNSIGNED_BYTE);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        
        glColor3f(0, 0, 0);
        pangolin::glDrawAxis(3);
         
        // draw poses
        glColor3f(0.5, 0.5, 0.5);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = cameraPaths_.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(cameraPaths_[i].x(), cameraPaths_[i].y(), cameraPaths_[i].z());
            glVertex3f(cameraPaths_[i+1].x(), cameraPaths_[i+1].y(), cameraPaths_[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator_.solverFlag_ == Estimator::SolverFlag::NON_LINEAR)
        {
            drawCamera(Twc_);
            
            glPointSize(5);
            glBegin(GL_POINTS);
            
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator_.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            
            glEnd();
        }
        
        imgTexture.Upload(show_img.data, GL_RGB, GL_UNSIGNED_BYTE);
        cam_img.Activate();
        glColor3f(1.0, 1.0, 1.0);
        imgTexture.RenderToViewportFlipY();
        
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    
    shutDownFlag_ = true;
    
    cout << "Drawer shutdown" << endl;
}

void System::drawCamera(pangolin::OpenGlMatrix& Twc)
{
    const float w = 0.4f;
    const float h = w*0.75;
    const float z = w*0.6;
    
    glPushMatrix();
    
    glMultMatrixd(Twc.m);

    glLineWidth(3);

    glColor3f(1.0f,0.0f,0.0f);

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
    
}
