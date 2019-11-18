#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "parameter.h"
#include "feature_manager.h"
#include "optimizer.h"
#include "backend/integration_base.h"
#include "initial/initial_alignment.h"

namespace myslam
{

class Estimator
{
  public:
    Estimator();
    ~Estimator();

    void clearState();
    void setParameter();
    void processIMU(double t, const Vector3d &linearAcc, const Vector3d &angularVel);
    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > &image,
                      double header);
    bool initialStructure();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    bool solveRelativeRT(const std::vector<std::pair<Vector3d, Vector3d> > &corres, Matrix3d &R, Vector3d &T);
    bool visualInitialAlign();
    void solveOdometry();
    void optimization();
    void vector2double();
    void double2vector();
    void slidingWindow();
    bool failureDetection();
    
    FeatureManager*     featureManager_;
    Optimizer*          optimizer_;
    std::map<double, ImageFrame> allFramesInWindow_;
    
    Eigen::Matrix2d     project_sqrt_info_;
    Matrix3d            ric_;
    Vector3d            tic_;
    double              td_;
    
    bool                firstImuFlag_;
    Vector3d            lastAcc_, lastGyr_;
    IntegrationBase*    preIntegrations_[WINDOW_SIZE + 1];
    IntegrationBase*    tmpPreIntegration_;
    int                 frameId_;
    int                 sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    
    std::vector<double>   dtBuf_[WINDOW_SIZE + 1];
    std::vector<Vector3d> linearAccBuf_[WINDOW_SIZE + 1];
    std::vector<Vector3d> angularVelBuf_[WINDOW_SIZE + 1];
    
    
    // parameters in sliding window under body frame
    Vector3d            g;
    Vector3d            Ps[WINDOW_SIZE + 1];
    Vector3d            Vs[WINDOW_SIZE + 1];
    Matrix3d            Rs[WINDOW_SIZE + 1];
    Vector3d            Bas[WINDOW_SIZE + 1];
    Vector3d            Bgs[WINDOW_SIZE + 1];
    
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };
    
    MarginalizationFlag marginalizationFlag_;
    double              headersInWindow_[WINDOW_SIZE + 1];
    
    SolverFlag          solverFlag_;
    double              initialTimestamp_;

    
    double              (*para_Pose)[SIZE_POSE];
    double              (*para_SpeedBias)[SIZE_SPEEDBIAS];
    double              (*para_Feature)[SIZE_FEATURE];
    double              para_Ex_Pose[SIZE_POSE];
    
    bool                relocalizeFlag_;
    bool                failureOccur_;
    Matrix3d            R0_before_slide_, lastR_, lastR0_;
    Vector3d            P0_before_slide_, lastP_, lastP0_;
    std::vector<Vector3d> key_poses;
    
    Vector3d            relocalize_t;
    Matrix3d            relocalize_r;
    
};

}

#endif
