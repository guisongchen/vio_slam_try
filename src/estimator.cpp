#include "estimator.h"
#include "utility.h"
#include "initial/initial_sfm.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <unordered_map>

namespace myslam
{
    
Estimator::Estimator()
{
    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
        preIntegrations_[i] = nullptr;
    
    for(auto &it: allFramesInWindow_)
        it.second.preIntegration_ = nullptr;
    
    tmpPreIntegration_ = nullptr;
    
    featureManager_ = new FeatureManager(Rs);
    
    para_Pose = new double[WINDOW_SIZE + 1][SIZE_POSE];
    para_SpeedBias = new double[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    para_Feature = new double[NUM_OF_F][SIZE_FEATURE];

    optimizer_ = new Optimizer(featureManager_);
    
    optimizer_->setOptimizeParam(para_Pose, para_SpeedBias, para_Feature, para_Ex_Pose, preIntegrations_);
    
    clearState();
}

Estimator::~Estimator()
{
    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
        delete preIntegrations_[i];
    
    delete[] para_Pose;
    delete[] para_SpeedBias;
    delete[] para_Feature;
    delete tmpPreIntegration_;
}


void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dtBuf_[i].clear();
        linearAccBuf_[i].clear();
        angularVelBuf_[i].clear();

        if (preIntegrations_[i] != nullptr)
            delete preIntegrations_[i];
        preIntegrations_[i] = nullptr;
    }

    tic_ = Vector3d::Zero();
    ric_ = Matrix3d::Identity();

    for (auto &it : allFramesInWindow_)
    {
        if (it.second.preIntegration_ != nullptr)
        {
            delete it.second.preIntegration_;
            it.second.preIntegration_ = nullptr;
        }
    }

    solverFlag_ = INITIAL;
    firstImuFlag_ = true,
    sum_of_back = 0;
    sum_of_front = 0;
    frameId_ = 0;
    initialTimestamp_ = 0;
    allFramesInWindow_.clear();
    td_ = TD;

    if (tmpPreIntegration_ != nullptr)
        delete tmpPreIntegration_;
    tmpPreIntegration_ = nullptr;

    featureManager_->clearState();

    failureOccur_ = false;
    relocalizeFlag_ = false;

    relocalize_t = Eigen::Vector3d(0, 0, 0);
    relocalize_r = Eigen::Matrix3d::Identity();
}

void Estimator::setParameter()
{
    tic_ = TIC[0];
    ric_ = RIC[0];
    cout << "Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    featureManager_->setRic(ric_);
    
    td_ = TD;
}

// dt = current_imu_time - last_imu_time
// linearAcc and angularVel are measurements directly from gyrscope
void Estimator::processIMU(double dt, const Vector3d &linearAcc, const Vector3d &angularVel)
{
    if (firstImuFlag_)
    {
        firstImuFlag_ = false;
        lastAcc_ = linearAcc;
        lastGyr_ = angularVel;
    }

    // check the current frame preIntegration exist or not, if not, create new one
    if (!preIntegrations_[frameId_])
    {
        preIntegrations_[frameId_] = new IntegrationBase{lastAcc_, lastGyr_, Bas[frameId_], Bgs[frameId_]};
    }
    
    // if frameId == 0, we can't do integration for only one element
    if (frameId_ != 0)
    {

        preIntegrations_[frameId_]->push_back(dt, linearAcc, angularVel);
        tmpPreIntegration_->push_back(dt, linearAcc, angularVel);
        
        dtBuf_[frameId_].push_back(dt);
        linearAccBuf_[frameId_].push_back(linearAcc);
        angularVelBuf_[frameId_].push_back(angularVel);
        
        int currId = frameId_;
        
        // calculate delatq and transform to Rotation Matrix
        Vector3d predictGyr = 0.5 * (lastGyr_ + angularVel) - Bgs[currId];
        Rs[currId] *= Utility::deltaQ(predictGyr * dt).toRotationMatrix();
        
        Vector3d lastLinearlizedAcc = Rs[currId] * (lastAcc_ - Bas[currId]) - g;
        Vector3d currLinearlizedAcc = Rs[currId] * (linearAcc - Bas[currId]) - g;
        Vector3d predictAcc = 0.5 * (lastLinearlizedAcc + currLinearlizedAcc);
        
        Ps[currId] += dt * Vs[currId] + 0.5 * dt * dt * predictAcc;
        Vs[currId] += dt * predictAcc;
    }
    
    // update measurement using current measurement to calculate next preIntegration
    lastAcc_ = linearAcc;
    lastGyr_ = angularVel;
}

void Estimator::processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > &image,
                             double header)
{
    if (featureManager_->addFeatureCheckParallax(frameId_, image, td_))
        marginalizationFlag_ = MARGIN_OLD;
    else
        marginalizationFlag_ = MARGIN_SECOND_NEW;

    headersInWindow_[frameId_] = header;

    ImageFrame imageframe(image, header);
    imageframe.preIntegration_ = tmpPreIntegration_;
    allFramesInWindow_.insert(std::make_pair(header, imageframe));
    
    // integration region: last frame ~ current frame
    tmpPreIntegration_ = new IntegrationBase{lastAcc_, lastGyr_, Bas[frameId_], Bgs[frameId_]};

    if (solverFlag_ == INITIAL)
    {
        // check if the window is full
        if (frameId_ == WINDOW_SIZE)
        {
            bool initResultFlag_ = false;
            
            // img pub gap is 0.1s
            if (ESTIMATE_EXTRINSIC != 2 && (header - initialTimestamp_) > 0.1)
            {
                initResultFlag_ = initialStructure();
                initialTimestamp_ = header;
            }
            
            if (initResultFlag_)
            {
                solverFlag_ = NON_LINEAR;
                
                solveOdometry();
                
                slidingWindow();
                
                featureManager_->removeFailures();
                
                cout << "Initialization finish!" << endl;
                
                // used in failureDetection in case big motion
                lastR_ = Rs[WINDOW_SIZE];
                lastP_ = Ps[WINDOW_SIZE];
                
                // used in statues recorrection, to recover yaw of frame 0 in window
                lastR0_ = Rs[0];
                lastP0_ = Ps[0];
            }
            else
                slidingWindow();
        }
        else
            frameId_++;
    }
    else
    {
        solveOdometry();

        if (failureDetection())
        {
            failureOccur_ = true;
            clearState();
            setParameter();

            return;
        }

        slidingWindow();
        
        featureManager_->removeFailures();

        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        lastR_ = Rs[WINDOW_SIZE];
        lastP_ = Ps[WINDOW_SIZE];
        lastR0_ = Rs[0];
        lastP0_ = Ps[0];
    }
}

bool Estimator::initialStructure()
{
    //check imu observibility, aka check linear accleration var(more fluctuation, good observibility)
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Vector3d sumAcc;
        
        // skip the first frame, why? first frame have no delta_v info, starting from second frame
        for (frame_it = allFramesInWindow_.begin(), frame_it++; frame_it != allFramesInWindow_.end(); frame_it++)
        {
            // frame_dt is the sum of all imu dt between two frames
            double frame_dt = frame_it->second.preIntegration_->sum_dt;
            Vector3d tmpAcc = frame_it->second.preIntegration_->delta_v / frame_dt;
            sumAcc += tmpAcc;
        }
        
        const int totalFrameNum = static_cast<int>(allFramesInWindow_.size()) - 1;
        
        Vector3d averAcc;
        averAcc = sumAcc / (1.0 * totalFrameNum);
        
        double var = 0;
        for (frame_it = allFramesInWindow_.begin(), frame_it++; frame_it != allFramesInWindow_.end(); frame_it++)
        {
            double frame_dt = frame_it->second.preIntegration_->sum_dt;
            Vector3d tmpAcc = frame_it->second.preIntegration_->delta_v / frame_dt;
            var += (tmpAcc - averAcc).transpose() * (tmpAcc - averAcc);
        }
        
        var = std::sqrt(var / (1.0 * totalFrameNum));
        
//         if (var < 0.25)
//             std::cerr << "IMU excitation not enough! var = " << var << endl;
    }
    
    // global sfm
    std::map<int, Vector3d> sfmTrackedPoints;
    std::vector<SFMFeature> sfmFeatures;
    
    for (auto &trackedFeature : featureManager_->trackedFeatures_)
    {
        SFMFeature tmpFeature;
        tmpFeature.triangulateState_ = false;
        tmpFeature.id_ = trackedFeature.featureId_;
        
        int frameId = trackedFeature.startFrameId_;        
        for (auto &covFeature : trackedFeature.covisibleFeatures_)
        {
            Vector3d p3dNormal = covFeature.point;
            tmpFeature.observation_.push_back(std::make_pair(frameId, Eigen::Vector2d(p3dNormal[0], p3dNormal[1])));
            
            frameId++;
        }
        
        sfmFeatures.push_back(tmpFeature);
    }
    
    Matrix3d Rrm;
    Vector3d trm;
    int refFrameId;
    
    // the one has more than 20 mathced points and 30 pixels parallax consider as reference frame
    // find reference frame and index in window
    if (!relativePose(Rrm, trm, refFrameId))
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }
    
    GlobalSFM sfm;
    const int frameNum = frameId_ + 1;
    Quaterniond quatsRefWorld[frameNum];
    Vector3d transRefWorld[frameNum];
    
    if (!sfm.construct(frameNum, quatsRefWorld, transRefWorld, refFrameId,
                       Rrm, trm, sfmFeatures, sfmTrackedPoints))
    {
        cout << "global SFM failed!" << endl;
        marginalizationFlag_ = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Vector3d>::iterator it;
    
    frame_it = allFramesInWindow_.begin();
    for (int i = 0; frame_it != allFramesInWindow_.end(); frame_it++)
    {
        // frame_it is in window, has been optimized, this should be always true
        if ((frame_it->first) == headersInWindow_[i])
        {
            frame_it->second.isKeyframe_ = true;
            frame_it->second.rotation_ = quatsRefWorld[i].toRotationMatrix() * ric_.transpose();
            
            // since we don't know scale yet, leave translation unchange for now
            frame_it->second.translation_ = transRefWorld[i];
            
            i++;
            
            continue;
        }
        
        if ((frame_it->first) > headersInWindow_[i])
            i++;
        
        // provide initial guess
        cv::Mat R_iw_mat, rvec, t_iw_mat, D, tmp_r;
        
        // set i-th frame pose in camera frame as initial value for frame_it
        Matrix3d R_iw_init = (quatsRefWorld[i].inverse()).toRotationMatrix();
        Vector3d t_iw_init = -R_iw_init * transRefWorld[i];
        
        cv::eigen2cv(R_iw_init, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(t_iw_init, t_iw_mat);

        frame_it->second.isKeyframe_ = false;
        std::vector<cv::Point3f> p3ds;
        std::vector<cv::Point2f> p2ds;
        
        for (auto &id_pts : frame_it->second.image_)
        {
            int feature_id = id_pts.first;
            
            for (auto &i_p : id_pts.second)
            {
                it = sfmTrackedPoints.find(feature_id);
                
                // feature was tracked, use optimized 3d poition
                if (it != sfmTrackedPoints.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f p3d(world_pts(0), world_pts(1), world_pts(2));
                    p3ds.push_back(p3d);
                    
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f p2d(img_pts(0), img_pts(1));
                    p2ds.push_back(p2d);
                }
            }
        }
        
        if (p3ds.size() < 6)
        {
            cout << "Not enough points for solve pnp pts_3_vector size " << p3ds.size() << endl;
            return false;
        }
        
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        
        // use r, t as initial value for useExtrinsicGuess = 1
        if (!cv::solvePnP(p3ds, p2ds, K, D, rvec, t_iw_mat, 1))
        {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        
        cv::Rodrigues(rvec, R_iw_mat);
        MatrixXd R_wi, R_iw;
        cv::cv2eigen(R_iw_mat, R_iw);
        R_wi = R_iw.transpose();
        
        MatrixXd t_iw;
        cv::cv2eigen(t_iw_mat, t_iw);
        Vector3d t_wi = R_wi * (-t_iw);
        
        // rotation: from camera frame to body frame
        frame_it->second.rotation_ = R_wi * ric_.transpose();
        frame_it->second.translation_ = t_wi;
    }
    
    if (visualInitialAlign())
        return true;
    else
    {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    VectorXd x;
    
    //solve scale
    bool result = VisualIMUAlignment(allFramesInWindow_, Bgs, g, x);
    
    if (!result)
    {
        cout << "solve g failed!" << endl;
        return false;
    }

    // update Ps, Rs data, from imu value to calculated value after triangulated in reference camera frame
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Matrix3d Ri = allFramesInWindow_[headersInWindow_[i]].rotation_;
        Vector3d Pi = allFramesInWindow_[headersInWindow_[i]].translation_;
        Ps[i] = Pi;
        Rs[i] = Ri;
        allFramesInWindow_[headersInWindow_[i]].isKeyframe_ = true;
    }
    
    for (auto &feature : featureManager_->trackedFeatures_)
    {
        if (feature.covisibleFeatures_.size() >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2)
            feature.estimatedDepth_ = -1.0;
    }

    //triangulat in body frame pose, with only ric and no tic, why no tic?
    Vector3d TIC_TMP;
    TIC_TMP.setZero();
    featureManager_->setRic(ric_);

    featureManager_->triangulate(Ps, TIC_TMP, ric_);

    for (int i = 0; i <= WINDOW_SIZE; i++)
        preIntegrations_[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    
    double s = (x.tail<1>())(0);
    for (int i = frameId_; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    
    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = allFramesInWindow_.begin(); frame_i != allFramesInWindow_.end(); frame_i++)
    {
        // update keyframe velocity
        if (frame_i->second.isKeyframe_)
        {
            kv++;
            Vs[kv] = frame_i->second.rotation_ * x.segment<3>(kv * 3);
        }
    }
    
    // update point depth with optimized scale
    for (auto &feature : featureManager_->trackedFeatures_)
    {
        if (!(feature.covisibleFeatures_.size() >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;
        
        feature.estimatedDepth_ *= s;
    }
    
    Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    
    // R0 = q21 = R_Gg
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    
    double yaw0 = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw0, 0, 0}) * R0;
    
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    
    // in fact g = R0*g = G
    g = R0 * g;

    // correct RVQ after compensate yaw angle of g 
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frameId_; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }

    return true;
}

// 
bool Estimator::relativePose(Matrix3d &Rrm, Vector3d &trm, int &refFrameId)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Vector3d, Vector3d> > correspondPoints;
        
        // find covFeature between i-th frame and current frame(WINDOW_SIZE)
        correspondPoints = featureManager_->getCorresponding(i, WINDOW_SIZE);
        
        const int correspondCnt = static_cast<int>(correspondPoints.size());
        
        if (correspondCnt > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            
            // check average distance on pixel plane
            for (int j = 0; j < correspondCnt; j++)
            {
                Vector2d pts_0(correspondPoints[j].first(0), correspondPoints[j].first(1));
                Vector2d pts_1(correspondPoints[j].second(0), correspondPoints[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = sum_parallax / correspondCnt;

            if (average_parallax * 460 > 30 && solveRelativeRT(correspondPoints, Rrm, trm))
            {
                refFrameId = i;
                return true;
            }
        }
    }
    return false;
}

bool Estimator::solveRelativeRT(const std::vector<std::pair<Vector3d, Vector3d> > &corres,
                                Matrix3d &Rotation_rm, Vector3d &Translation_rm)
{
    if (corres.size() >= 15) // only if corres size over 20, then call this function. redundancy.
    {
        std::vector<cv::Point2f> RefPoints, MatchPoints;
        for (int i = 0; i < int(corres.size()); i++)
        {
            RefPoints.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            MatchPoints.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        
        cv::Mat mask;
        
        cv::Mat E = cv::findFundamentalMat(RefPoints, MatchPoints, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot_mr, trans_mr;
        int inlier_cnt = cv::recoverPose(E, RefPoints, MatchPoints, cameraMatrix, rot_mr, trans_mr, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans_mr.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot_mr.at<double>(i, j);
        }

        Rotation_rm = R.transpose();
        Translation_rm = -R.transpose() * T;
        
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    
    return false;
}

void Estimator::solveOdometry()
{
    if (frameId_ < WINDOW_SIZE)
        return;
    
    if (solverFlag_ == NON_LINEAR)
    {
        featureManager_->triangulate(Ps, tic_, ric_);
        
        optimization();
    }
}

void Estimator::optimization()
{
    vector2double();
    
    optimizer_->solveProblem();
    
    double2vector();

    if (marginalizationFlag_ == MARGIN_OLD)
    {
        
        vector2double();
        
        optimizer_->margOldFrame();
        
        std::unordered_map<long, double *> shiftedAddress;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            shiftedAddress[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            shiftedAddress[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        
        shiftedAddress[reinterpret_cast<long>(para_Ex_Pose)] = para_Ex_Pose;
        
    }

    // MARGIN_SECOND_NEW
    // only remove corresponding visual measurement(pose), keep IMU info(speed and bias)
    else
    {
        if (optimizer_->lastMargFlag_)
        {
            
            vector2double();
            
            optimizer_->margNewFrame();

            // adress: address0     address1     address2     address3     address4
            // before: para_Pose[0] para_Pose[1] para_Pose[2] para_Pose[3] para_Pose[4]
            // after:  para_Pose[0] para_Pose[1] para_Pose[2] para_Pose[4] -Avaliable-
            
            std::unordered_map<long, double *> shiftedAddress;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    shiftedAddress[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    shiftedAddress[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    shiftedAddress[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    shiftedAddress[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            
            shiftedAddress[reinterpret_cast<long>(para_Ex_Pose)] = para_Ex_Pose;
        }
    }
    
}


void Estimator::slidingWindow()
{
    if (marginalizationFlag_ == MARGIN_OLD)
    {
        R0_before_slide_ = Rs[0];
        P0_before_slide_ = Ps[0];
        
        double t_0 = headersInWindow_[0];
        
        if (frameId_ == WINDOW_SIZE)
        {
            // frame 0 was marginalized, slide window leftward
            // we get value 1~WINDOW_SIZE move to postion 0~WINDOW_SIZE, left postion WINDOW_SIZE unchange
            
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                std::swap(preIntegrations_[i], preIntegrations_[i + 1]);
                dtBuf_[i].swap(dtBuf_[i + 1]);
                linearAccBuf_[i].swap(linearAccBuf_[i + 1]);
                angularVelBuf_[i].swap(angularVelBuf_[i + 1]);
                headersInWindow_[i] = headersInWindow_[i + 1];
                
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            
            headersInWindow_[WINDOW_SIZE] = headersInWindow_[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete preIntegrations_[WINDOW_SIZE];
            preIntegrations_[WINDOW_SIZE] = new IntegrationBase{lastAcc_, lastGyr_, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            
            dtBuf_[WINDOW_SIZE].clear();
            linearAccBuf_[WINDOW_SIZE].clear();
            angularVelBuf_[WINDOW_SIZE].clear();

            std::map<double, ImageFrame>::iterator it_0;
            
            it_0 = allFramesInWindow_.find(t_0);
            delete it_0->second.preIntegration_;
            it_0->second.preIntegration_ = nullptr;
            
            for (std::map<double, ImageFrame>::iterator it = allFramesInWindow_.begin(); it != it_0; ++it)
            {
                if (it->second.preIntegration_)
                    delete it->second.preIntegration_;
                
                it->second.preIntegration_ = nullptr;
            }
            
            allFramesInWindow_.erase(allFramesInWindow_.begin(), it_0);
            allFramesInWindow_.erase(t_0);

            sum_of_back++;

            if (solverFlag_ == NON_LINEAR)
            {
                // pose before and after slide
                Matrix3d R0, R1;
                Vector3d P0, P1;
                
                // from camera frame to body frame
                R0 = R0_before_slide_ * ric_;
                R1 = Rs[0] * ric_;
                
                P0 = P0_before_slide_ + R0_before_slide_ * tic_;
                P1 = Ps[0] + Rs[0] * tic_;
                
                featureManager_->removeBackShiftDepth(R0, P0, R1, P1);
            }
            else
                featureManager_->removeOld();
        }
    }
    else
    {
        if (frameId_ == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dtBuf_[frameId_].size(); i++)
            {
                double tmp_dt = dtBuf_[frameId_][i];
                Vector3d tmp_linear_acceleration = linearAccBuf_[frameId_][i];
                Vector3d tmp_angular_velocity = angularVelBuf_[frameId_][i];

                preIntegrations_[frameId_ - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dtBuf_[frameId_ - 1].push_back(tmp_dt);
                linearAccBuf_[frameId_ - 1].push_back(tmp_linear_acceleration);
                angularVelBuf_[frameId_ - 1].push_back(tmp_angular_velocity);
            }

            headersInWindow_[frameId_ - 1] = headersInWindow_[frameId_];
            Ps[frameId_ - 1] = Ps[frameId_];
            Vs[frameId_ - 1] = Vs[frameId_];
            Rs[frameId_ - 1] = Rs[frameId_];
            Bas[frameId_ - 1] = Bas[frameId_];
            Bgs[frameId_ - 1] = Bgs[frameId_];

            delete preIntegrations_[WINDOW_SIZE];
            preIntegrations_[WINDOW_SIZE] = new IntegrationBase{lastAcc_, lastGyr_, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dtBuf_[WINDOW_SIZE].clear();
            linearAccBuf_[WINDOW_SIZE].clear();
            angularVelBuf_[WINDOW_SIZE].clear();

            sum_of_front++;
            featureManager_->removeSecondNew();
        }
    }
}

void Estimator::vector2double()
{
    // PVQ and bias
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        
        Quaterniond q{Rs[i]};
        
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    
    // extrinsic between imu and camera
    para_Ex_Pose[0] = tic_.x();
    para_Ex_Pose[1] = tic_.y();
    para_Ex_Pose[2] = tic_.z();
    
    Quaterniond q{ric_};
    
    para_Ex_Pose[3] = q.x();
    para_Ex_Pose[4] = q.y();
    para_Ex_Pose[5] = q.z();
    para_Ex_Pose[6] = q.w();
    
    // inverse depth info
    VectorXd invDepths = featureManager_->getInvDepthVector();
    
    for (int i = 0; i < featureManager_->getFeatureCount(); i++)
        para_Feature[i][0] = invDepths(i);
}

void Estimator::double2vector()
{
    // rotation and translation vector before optimization
    Vector3d origin_r0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failureOccur_)
    {
        origin_r0 = Utility::R2ypr(lastR0_);
        origin_P0 = lastP0_;
        failureOccur_ = false;
    }
    
    // rotation after optimization
    Vector3d optimized_r0 = Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                                     para_Pose[0][4], para_Pose[0][5]).toRotationMatrix());

    double yawDrift = origin_r0.x() - optimized_r0.x();
    Matrix3d rotDrift = Utility::ypr2R(Vector3d(yawDrift, 0, 0));
    
    if (abs(abs(origin_r0.y()) - 90) < 1.0 || abs(abs(optimized_r0.y()) - 90) < 1.0)
    {
        cout << "euler singular point!" << endl;
        rotDrift = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    // update PVQB vectors using optimized value and drift compensation
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = rotDrift * Quaterniond(para_Pose[i][6], para_Pose[i][3],
                                       para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rotDrift * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
                                    
        Vs[i] = rotDrift * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);
        
        // bias is unaffected by drift
        Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);
        Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
    }

    // update extrinsic parameter
    tic_ = Vector3d(para_Ex_Pose[0], para_Ex_Pose[1], para_Ex_Pose[2]);
    ric_ = Quaterniond(para_Ex_Pose[6], para_Ex_Pose[3], para_Ex_Pose[4], para_Ex_Pose[5]).toRotationMatrix();

    // update depth
    VectorXd invDepths = featureManager_->getInvDepthVector();
    
    for (int i = 0; i < featureManager_->getFeatureCount(); i++)
        invDepths(i) = para_Feature[i][0];
    
    featureManager_->setDepth(invDepths);
}

bool Estimator::failureDetection()
{
    if (featureManager_->lastTrackNum_ < 2)
    {
        cout << "Little feature: " << featureManager_->lastTrackNum_ << endl;
        return true;
    }
    
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        cout <<"Big IMU acc bias estimation: " << Bas[WINDOW_SIZE].norm() << endl;
        return true;
    }
    
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        cout <<"Big IMU gyr bias estimation: " << Bgs[WINDOW_SIZE].norm() << endl;
        return true;
    }
    
    Vector3d &currP = Ps[WINDOW_SIZE];
    
    if ((currP - lastP_).norm() > 5)
    {
        cout << "Big translation" << endl;
        return true;
    }
    
    if (abs(currP.z() - lastP_.z()) > 1)
    {
        cout<< "Big z translation" << endl;
        return true;
    }
    
    Matrix3d &currR = Rs[WINDOW_SIZE];
    Matrix3d deltaR = currR.transpose() * lastR_;
    Quaterniond deltaQ(deltaR);
    double deltaTheta = acos(deltaQ.w()) * 2.0 / 3.14 * 180.0;
    
    if (deltaTheta > 50)
    {
        cout << "Big delta angle" << endl;
        return true;
    }
    
    return false;
}
    
}
