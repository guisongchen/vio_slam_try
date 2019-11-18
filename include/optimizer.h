#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "backend/integration_base.h"
#include "feature_manager.h"
#include "parameter.h"
#include "backend/vertex_types.h"
#include "backend/edge_types.h"

namespace myslam
{

class Optimizer 
{
public:
    
    Optimizer(FeatureManager* featureManager);
    
    void setOptimizeParam(double _para_Pose[][SIZE_POSE], double _para_SpeedBias[][SIZE_SPEEDBIAS],
                          double _para_Feature[][SIZE_FEATURE], double* _para_Ex_Pose,
                          IntegrationBase** _preIntegration);
    
    void solveProblem();
    void margOldFrame();
    void margNewFrame();
    
    bool                lastMargFlag_;
    
private:
    
    FeatureManager*     featureManager_;
    IntegrationBase**   preIntegrations;
    
    double              (*para_Pose)[SIZE_POSE];
    double              (*para_SpeedBias)[SIZE_SPEEDBIAS];
    double              (*para_Feature)[SIZE_FEATURE];
    double*             para_Ex_Pose;
    
    Eigen::Matrix2d     projectionInfoMatrix_;
    
    Eigen::MatrixXd     H_prior_;
    Eigen::VectorXd     b_prior_;
    Eigen::VectorXd     err_prior_;
    Eigen::MatrixXd     Jprior_inv_;
    
};
    

    
}

#endif
