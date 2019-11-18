#include "backend/vertex_types.h"
#include "sophus/se3.h"

namespace myslam
{

namespace backend
{
    
void VertexPose::SetParameterData(double* param)
{
    Eigen::Map<Eigen::Matrix<double, 7, 1> > param_data(param);
    parameters_ = param_data;
}

void VertexPose::Plus(const Eigen::VectorXd &delta)
{
    parameters_.head<3>() += delta.head<3>();
    
    Eigen::Quaterniond q(parameters_[6], parameters_[3], parameters_[4], parameters_[5]);
    q = q * Sophus::SO3::exp(Eigen::Vector3d(delta[3], delta[4], delta[5])).unit_quaternion();
    q.normalized();
    
    parameters_[3] = q.x();
    parameters_[4] = q.y();
    parameters_[5] = q.z();
    parameters_[6] = q.w();
}

void VertexSpeedBias::SetParameterData(double* param)
{
    Eigen::Map<Eigen::Matrix<double, 9, 1> > param_data(param);
    parameters_ = param_data;
}

void VertexInverseDepth::SetParameterData(double* param)
{
    parameters_[0]= param[0];
}

void VertexPointXYZ::SetParameterData(double* param)
{
    Eigen::Map<Eigen::Vector3d> param_data(param);
    parameters_ = param_data;
}

} // namespace backend

} // namespace myslam

