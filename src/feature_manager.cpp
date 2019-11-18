#include "feature_manager.h"

namespace myslam
{
    
FeatureManager::FeatureManager(Matrix3d Rs[])
    : Rs_(Rs)
{
        ric_.setIdentity();
}
    
void FeatureManager::setRic(Matrix3d ric)
{
        ric_ = ric;
}

bool FeatureManager::addFeatureCheckParallax(int frameId,
    const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > &image, double td)
{
    lastTrackNum_ = 0;
    
    for (auto &id_pts : image)
    {
        int featureId = id_pts.first;
        
        // use date from camera 0
        Eigen::Matrix<double, 7, 1> currXYZUVVxVy = id_pts.second[0].second;
        
        CovisibleFeature currCovFeature(currXYZUVVxVy, td);

        auto it = find_if(trackedFeatures_.begin(), trackedFeatures_.end(),
                          [featureId] (const Feature &it) { return it.featureId_ == featureId; });

        // if not exist, added and update
        if (it == trackedFeatures_.end())
        {
            // init Feature(including startFrameId_)
            trackedFeatures_.push_back(Feature(featureId, frameId));
            trackedFeatures_.back().covisibleFeatures_.push_back(currCovFeature);
        }
        
        // if find match, which means this feature was tracked before, update CovisibleFeature info
        else if (it->featureId_ == featureId)
        {
            it->covisibleFeatures_.push_back(currCovFeature);
            lastTrackNum_++;
        }
    }
    
    // if frame in current window less than 2, or tracked feature less than 20, return true
    if (frameId < 2 || lastTrackNum_ < 20)
        return true;

    double parallaxSum = 0.0;
    int parallaxNum = 0;
    
    for (auto &trackedFeature : trackedFeatures_)
    {
        const int startFrameId = trackedFeature.startFrameId_;
        const int trackedFrameNum = static_cast<int>(trackedFeature.covisibleFeatures_.size());
        const int trackEndId = startFrameId + trackedFrameNum - 1;

        if (startFrameId <= frameId - 2 && trackEndId >= frameId - 1)
        {
            parallaxSum += computeParallax(trackedFeature, frameId);
            parallaxNum++;
        }
    }

    // no qualified covFeature, tracking is weaken, consider current as frame keyframe
    if (parallaxNum == 0)
        return true;
    else
        return (parallaxSum / parallaxNum) >= MIN_PARALLAX;
}

// point i and j distance on normalized plane 
double FeatureManager::computeParallax(const Feature &trackedFeature, int frameId)
{ 
    const CovisibleFeature &frame_i = trackedFeature.covisibleFeatures_[frameId - 2 - trackedFeature.startFrameId_];
    const CovisibleFeature &frame_j = trackedFeature.covisibleFeatures_[frameId - 1 - trackedFeature.startFrameId_];

    double ans = 0.0;
    
    Vector3d p_j = frame_j.point;
    Vector3d p_i = frame_i.point;

    ans = (p_i - p_j).norm();
    
    return ans;
}

std::vector<std::pair<Vector3d, Vector3d> > FeatureManager::getCorresponding(int queryRefFrameId,
                                                                             int queryMatchFrameId)
{
    std::vector<std::pair<Vector3d, Vector3d> > correspondPonits;
    
    for (auto &trackedFeature : trackedFeatures_)
    {
        // current feature tracked region: from startFrameId to endFrameId
        int startFrameId = trackedFeature.startFrameId_;
        int endFrameId = startFrameId + trackedFeature.covisibleFeatures_.size() - 1;
        
        if (startFrameId <= queryRefFrameId && endFrameId >= queryMatchFrameId)
        {

            int refIdx = queryRefFrameId - trackedFeature.startFrameId_;
            int matchIdx = queryMatchFrameId - trackedFeature.startFrameId_;
            Vector3d a = trackedFeature.covisibleFeatures_[refIdx].point;
            Vector3d b = trackedFeature.covisibleFeatures_[matchIdx].point;
            
            // two 3d point info of same covisible feature, will be used for SFM
            correspondPonits.push_back(std::make_pair(a, b));
        }
    }
    
    return correspondPonits;
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : trackedFeatures_)
    {
        it.covisibleNum_ = it.covisibleFeatures_.size();
        if (it.covisibleNum_ >= 2 && it.startFrameId_ < WINDOW_SIZE - 2)
            cnt++;
    }
    
    return cnt;
}

VectorXd FeatureManager::getInvDepthVector()
{
    VectorXd invDepthVector(getFeatureCount());
    
    int featureIndex = 0;
    for (auto &feature : trackedFeatures_)
    {
        feature.covisibleNum_ = feature.covisibleFeatures_.size();
        
        // at least be observed by 2 frames, not far away from current frame(window-1)
        if (!(feature.covisibleNum_ >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;

        invDepthVector(featureIndex++) = 1.0 / feature.estimatedDepth_;
    }
    
    return invDepthVector;
}

void FeatureManager::setDepth(const VectorXd &invDepths)
{
    int featureIndex = 0;
    for (auto &feature : trackedFeatures_)
    {
        feature.covisibleNum_ = feature.covisibleFeatures_.size();
        if (!(feature.covisibleNum_ >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;

        feature.estimatedDepth_ = 1.0 / invDepths(featureIndex);
        
        featureIndex++;
        
        if (feature.estimatedDepth_ < 0)
            feature.solveFlag_ = 2; // solve failed
        else
            feature.solveFlag_ = 1; // solve succed
    }
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d &tic, Matrix3d &ric)
{
    for (auto &feature : trackedFeatures_)
    {

        if (!(feature.covisibleFeatures_.size() >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;

        // skip if already calculated
        if (feature.estimatedDepth_ > 0)
            continue;
        
        int imu_i = feature.startFrameId_;
        int imu_j = imu_i - 1;

        //! Ax=0
        assert(NUM_OF_CAM == 1);
        
        Eigen::MatrixXd svd_A(2 * feature.covisibleFeatures_.size(), 4);
        int svdRowNum = 0;

        // from body frame to camera frame
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs_[imu_i] * tic;
        Eigen::Matrix3d R0 = Rs_[imu_i] * ric;
        
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &covFeature : feature.covisibleFeatures_)
        {
            imu_j++;

            
            Eigen::Vector3d t1 = Ps[imu_j] + Rs_[imu_j] * tic;
            Eigen::Matrix3d R1 = Rs_[imu_j] * ric;

            // solve relative rots and trans from j to i in camera frame
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;

            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;

            Eigen::Vector3d f = covFeature.point.normalized();
            svd_A.row(svdRowNum++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svdRowNum++) = f[1] * P.row(2) - f[2] * P.row(1);
        }
        
        assert(svdRowNum == svd_A.rows());
        
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svdDepth = svd_V[2] / svd_V[3];

        feature.estimatedDepth_ = svdDepth;

        if (feature.estimatedDepth_ < 0.1)
            feature.estimatedDepth_ = INIT_DEPTH;

    }
}

// input: frame pose in body frame
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = trackedFeatures_.begin(), it_next = trackedFeatures_.begin() ; it != trackedFeatures_.end(); it = it_next)
    {
        it_next++;
        
        if (it->startFrameId_ != 0)
            it->startFrameId_--;
        else
        {
            Eigen::Vector3d uv_i = it->covisibleFeatures_[0].point;  
            it->covisibleFeatures_.erase(it->covisibleFeatures_.begin());
            
            if (it->covisibleFeatures_.size() < 2)
            {
                trackedFeatures_.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimatedDepth_;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimatedDepth_ = dep_j;
                else
                    it->estimatedDepth_ = INIT_DEPTH;
            }
        }
    }
}


void FeatureManager::removeOld()
{
    for (auto it = trackedFeatures_.begin(), it_next = trackedFeatures_.begin() ; it != trackedFeatures_.end(); it = it_next)
    {
        it_next++;
        
        if (it->startFrameId_ != 0)
            it->startFrameId_--;
        else
        {
            // if startFrameId == 0, current feature was first observed by marginalized frame
            // remove covFeature of marginalized frame 
            it->covisibleFeatures_.erase(it->covisibleFeatures_.begin());
            
            if (it->covisibleFeatures_.empty())
                trackedFeatures_.erase(it);
        }
    }
}



void FeatureManager::removeSecondNew()
{
    
    const int marginalizedId = WINDOW_SIZE - 1;
    
    for (auto it = trackedFeatures_.begin(), it_next = trackedFeatures_.begin() ; it != trackedFeatures_.end(); it = it_next)
    {
        it_next++;
        
        if (it->startFrameId_ == WINDOW_SIZE)
            it->startFrameId_--;
        else
        {
            const int endFrameId = it->startFrameId_ + it->covisibleFeatures_.size() -1;
            if (endFrameId < marginalizedId)
                continue;
            
            // remove covFeature of marginalized frame
            const int j = marginalizedId - it->startFrameId_;
            it->covisibleFeatures_.erase(it->covisibleFeatures_.begin() + j);
            
            if (it->covisibleFeatures_.empty())
                trackedFeatures_.erase(it);
                
        }
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = trackedFeatures_.begin(), it_next = trackedFeatures_.begin(); it != trackedFeatures_.end(); it = it_next)
    {
        it_next++;
        if (it->solveFlag_ == 2)
            trackedFeatures_.erase(it);
    }
}

void FeatureManager::clearState()
{
    trackedFeatures_.clear();
}
    
}
