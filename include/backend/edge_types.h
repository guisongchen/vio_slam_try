#ifndef MYSLAM_BACKEND_EDGE_TYPES_H
#define MYSLAM_BACKEND_EDGE_TYPES_H

#include "base_edge.h"
#include "base_vertex.h"
#include "integration_base.h"

namespace myslam
{
namespace backend
{

//--------------------IMU reprojection-----------------------
    
class EdgeImu : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeImu(IntegrationBase* pre_integration) 
        : Edge(15, 4), pre_integration_(pre_integration) {}

    virtual void ComputeResidual() override;
    virtual void ComputeJacobians() override;

private:
    IntegrationBase*    pre_integration_;

};

//--------------------invdepth reprojection-----------------------

class EdgeReprojection : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeReprojection(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j) : Edge(2, 4)
    {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }
    
    virtual void ComputeResidual() override;
    virtual void ComputeJacobians() override;

private:
    Eigen::Vector3d pts_i_, pts_j_;
};


//--------------------XYZ reprojection-----------------------

class EdgeReprojectXYZ : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeReprojectXYZ(const Eigen::Vector2d& measurement) : Edge(2, 2), measurement_(measurement) {}
        
    void SetCameraIntrisic(const double fx, const double fy, const double cx, const double cy);
    Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d& point) const;
    
    virtual void ComputeResidual() override;
    virtual void ComputeJacobians() override;

private:
    Eigen::Vector2d measurement_;
    double fx_, fy_, cx_, cy_;
};

} // namespace backend
} // namespace myslam
#endif

