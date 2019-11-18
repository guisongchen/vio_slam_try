#include "optimizer.h"
#include "backend/loss_function.h"
#include "backend/problem.h"

#include <unordered_map>

namespace myslam
{

Optimizer::Optimizer(FeatureManager* featureManager) : featureManager_(featureManager)
{
    projectionInfoMatrix_ = (FOCAL_LENGTH / 1.5) * (FOCAL_LENGTH / 1.5) * Eigen::Matrix2d::Identity();
    lastMargFlag_ = false;
}

void Optimizer::setOptimizeParam(double _para_Pose[][SIZE_POSE], double _para_SpeedBias[][SIZE_SPEEDBIAS],
                                 double _para_Feature[][SIZE_FEATURE], double* _para_Ex_Pose,
                                 IntegrationBase** _preIntegration)
{    
    para_Pose = _para_Pose;
    para_SpeedBias = _para_SpeedBias;
    para_Feature = _para_Feature;
    para_Ex_Pose = _para_Ex_Pose;
    preIntegrations = _preIntegration;
}
      

void Optimizer::solveProblem()
{
    
    backend::LossFunction* loss_function;
    loss_function = new backend::CauchyLoss(1.0);
    
    backend::Problem* problem(new backend::Problem());
    backend::AlgorithmLM* algorithmLM(new backend::AlgorithmLM(problem));
    problem->SetAlgorithm(algorithmLM);
    problem->SetReport(false);
    
    std::vector<std::shared_ptr<backend::VertexPose> > vertexCams;
    std::vector<std::shared_ptr<backend::VertexSpeedBias> > vertexVBs;
    int poseDimension = 0;

    // add extrinsic
    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    vertexExt->SetParameterData(para_Ex_Pose);
    if (!ESTIMATE_EXTRINSIC)
        vertexExt->SetFixed();
    
    problem->AddVertex(vertexExt);
    poseDimension += vertexExt->LocalDimension();

    // add frame PVQB
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        vertexCam->SetParameterData(para_Pose[i]);
        vertexCams.push_back(vertexCam);
        problem->AddVertex(vertexCam);
        poseDimension += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        vertexVB->SetParameterData(para_SpeedBias[i]);
        vertexVBs.push_back(vertexVB);
        problem->AddVertex(vertexVB);
        poseDimension += vertexVB->LocalDimension();
    }

    // IMU constrain
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        
        if (preIntegrations[j]->sum_dt > 10.0)
            continue;

        std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(preIntegrations[j]));
        
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams[i]);
        edge_vertex.push_back(vertexVBs[i]);
        edge_vertex.push_back(vertexCams[j]);
        edge_vertex.push_back(vertexVBs[j]);
        
        imuEdge->SetVertex(edge_vertex);
        problem->AddEdge(imuEdge);
    }

    // Visual constrain
    std::vector<std::shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;

    int feature_index = -1;
    for (auto &feature : featureManager_->trackedFeatures_)
    {
        if (!(feature.covisibleFeatures_.size() >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        const int imu_i = feature.startFrameId_;
        int imu_j = imu_i - 1;
        Vector3d pts_i = feature.covisibleFeatures_[0].point;

        std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
        verterxPoint->SetParameterData(para_Feature[feature_index]);
        
        problem->AddVertex(verterxPoint);
        vertexPt_vec.push_back(verterxPoint);

        for (auto &covFeature : feature.covisibleFeatures_)
        {
            imu_j++;
            if (imu_i == imu_j)
                continue;

            Vector3d pts_j = covFeature.point;
            std::shared_ptr<backend::EdgeReprojection> projectionEdge(new backend::EdgeReprojection(pts_i, pts_j));
            
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(verterxPoint);
            edge_vertex.push_back(vertexCams[imu_i]);
            edge_vertex.push_back(vertexCams[imu_j]);
            edge_vertex.push_back(vertexExt);
            
            projectionEdge->SetVertex(edge_vertex);
            projectionEdge->SetInformation(projectionInfoMatrix_);
            projectionEdge->SetLossFunction(loss_function);
            
            problem->AddEdge(projectionEdge);
        }
    }

    // prior residual constrain
    if (lastMargFlag_)
        problem->AddPriorInfo(H_prior_, b_prior_);

    problem->SolveProblem(10);

    // TODO why not update H_prior_?? ==> FEJ
    // update bprior_,  Hprior_ do not need update
    if (lastMargFlag_)
        b_prior_ = problem->GetbPrior();

    // update parameter after optimization
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Eigen::VectorXd pose = vertexCams[i]->Parameters();
        for (int j = 0; j < 7; ++j)
            para_Pose[i][j] = pose[j];

        Eigen::VectorXd speedBias = vertexVBs[i]->Parameters();
        for (int j = 0; j < 9; ++j)
            para_SpeedBias[i][j] = speedBias[j];
    }

    for (int i = 0; i < vertexPt_vec.size(); ++i)
    {
        Eigen::VectorXd invDepth = vertexPt_vec[i]->Parameters();
        para_Feature[i][0] = invDepth[0];
    }
    
    delete loss_function;
    delete algorithmLM;
    delete problem;
}

void Optimizer::margOldFrame()
{
    backend::LossFunction* loss_function;
    loss_function = new backend::CauchyLoss(1.0);

    backend::Problem* problem(new backend::Problem());
    
    std::vector<std::shared_ptr<backend::VertexPose>> vertexCams;
    std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVBs;
    int pose_dim = 0;
    
    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    vertexExt->SetParameterData(para_Ex_Pose);
    problem->AddVertex(vertexExt);
    pose_dim += vertexExt->LocalDimension();

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        vertexCam->SetParameterData(para_Pose[i]);
        vertexCams.push_back(vertexCam);
        problem->AddVertex(vertexCams[i]);
        pose_dim += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        vertexVB->SetParameterData(para_SpeedBias[i]);        
        vertexVBs.push_back(vertexVB);
        problem->AddVertex(vertexVBs[i]);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU
    if (preIntegrations[1]->sum_dt < 10.0)
    {
        std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(preIntegrations[1]));
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        
        edge_vertex.push_back(vertexCams[0]);
        edge_vertex.push_back(vertexVBs[0]);
        edge_vertex.push_back(vertexCams[1]);
        edge_vertex.push_back(vertexVBs[1]);
        
        imuEdge->SetVertex(edge_vertex);
        problem->AddEdge(imuEdge);
    }

    // Visual Factor
    int feature_index = -1;
    for (auto &feature : featureManager_->trackedFeatures_)
    {
        if (!(feature.covisibleFeatures_.size() >= 2 && feature.startFrameId_ < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        const int imu_i = feature.startFrameId_;
        int imu_j = imu_i - 1;
        
        if (imu_i != 0)
            continue;

        Vector3d pts_i = feature.covisibleFeatures_[0].point;

        std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
        verterxPoint->SetParameterData(para_Feature[feature_index]);
        
        problem->AddVertex(verterxPoint);

        for (auto &covFeature : feature.covisibleFeatures_)
        {
            imu_j++;
            if (imu_i == imu_j)
                continue;

            Vector3d pts_j = covFeature.point;
            std::shared_ptr<backend::EdgeReprojection> projectionEdge(new backend::EdgeReprojection(pts_i, pts_j));
            
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(verterxPoint);
            edge_vertex.push_back(vertexCams[imu_i]);
            edge_vertex.push_back(vertexCams[imu_j]);
            edge_vertex.push_back(vertexExt);
            
            projectionEdge->SetVertex(edge_vertex);
            projectionEdge->SetInformation(projectionInfoMatrix_);
            projectionEdge->SetLossFunction(loss_function);
            
            problem->AddEdge(projectionEdge);
        }
    }

 
    if (lastMargFlag_)
        problem->AddPriorInfo(H_prior_, b_prior_);
    else
        lastMargFlag_ = true;

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    marg_vertex.push_back(vertexCams[0]);
    marg_vertex.push_back(vertexVBs[0]);
    
    problem->Marginalize(marg_vertex, pose_dim);
    H_prior_ = problem->GetHessianPrior();
    b_prior_ = problem->GetbPrior();

    delete loss_function;
    delete problem;
}


void Optimizer::margNewFrame()
{
    backend::Problem* problem(new backend::Problem());
    
    std::vector<std::shared_ptr<backend::VertexPose>> vertexCams;
    std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVBs;
    int pose_dim = 0;

    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    vertexExt->SetParameterData(para_Ex_Pose);
    problem->AddVertex(vertexExt);
    pose_dim += vertexExt->LocalDimension();

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        vertexCam->SetParameterData(para_Pose[i]);
        vertexCams.push_back(vertexCam);
        problem->AddVertex(vertexCams[i]);
        pose_dim += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        vertexVB->SetParameterData(para_SpeedBias[i]);
        vertexVBs.push_back(vertexVB);
        problem->AddVertex(vertexVBs[i]);
        pose_dim += vertexVB->LocalDimension();
    }

    if (lastMargFlag_)
        problem->AddPriorInfo(H_prior_, b_prior_);
    else
        lastMargFlag_ = true;

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    marg_vertex.push_back(vertexCams[WINDOW_SIZE - 1]);
    marg_vertex.push_back(vertexVBs[WINDOW_SIZE - 1]);
    
    problem->Marginalize(marg_vertex, pose_dim);
    H_prior_ = problem->GetHessianPrior();
    b_prior_ = problem->GetbPrior();

    delete problem;

}

}
