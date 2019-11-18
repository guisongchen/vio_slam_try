#include "backend/base_edge.h"
#include <Eigen/Dense>

namespace myslam
{
namespace backend
{

unsigned long global_edge_id;
    
Edge::Edge(int residual_dimension, int num_verticies)
{
    residual_.resize(residual_dimension, 1);
    jacobians_.resize(num_verticies);
    
    id_ = global_edge_id++;

    information_.resize(residual_dimension, residual_dimension);
    information_.setIdentity();

    lossfunction_ = static_cast<LossFunction*>(nullptr);
}

bool Edge::AddVertex(std::shared_ptr<Vertex> vertex)
{
    verticies_.emplace_back(vertex);
    return true;
}

bool Edge::SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices)
{
    verticies_ = vertices;
    return true;
}

void Edge::SetInformation(const Eigen::MatrixXd &information)
{
    information_ = information;
    sqrt_information_ = Eigen::LLT<Eigen::MatrixXd>(information_).matrixL().transpose();
}

double Edge::Chi2() const
{
    return residual_.transpose() * information_ * residual_;
}

double Edge::RobustChi2() const
{

    double e2 = this->Chi2();
    
    if(lossfunction_)
    {
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        e2 = rho[0];
    }
    
    return e2;
}

// NOTE Gauss-Newton Hessian(see ceres lost_function)
// H = J^T * (rho[1] + 2 * rho[2] * residual * residual^T) * J
//
// with info:
// H = J^T * (rho[1] + 2 * rho[2] * (residual*info*residual^T)) * info * J
//   = J^T * (rho[1]*info + 2 * rho[2] * (info*residual) * (info*residual)^T) * J
// ==> robustInfo = rho[1]*info + 2 * rho[2] * (info*residual) * (info*residual)^T
//
// define weight_err = info*residual
// ==> robustInfo = rho[1]*info + 2 * rho[2] * weight_err * weight_err^T

void Edge::RobustInfo(double &drho, Eigen::MatrixXd &info) const
{
    if(lossfunction_)
    {
        /// robust_info = rho[1] * information_ + information_ * r * r^T * information_

        double e2 = this->Chi2();
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        Eigen::VectorXd weight_err = information_ * residual_;

        Eigen::MatrixXd robust_info(information_.rows(), information_.cols());
        robust_info.setIdentity();
        robust_info *= rho[1] * information_;
        
        if(rho[1] + 2 * rho[2] * e2 > 0.)
        {
            robust_info += 2 * rho[2] * weight_err * weight_err.transpose(); 
        }

        info = robust_info;
        drho = rho[1];
    }
    else
    {
        drho = 1.0;
        info = information_;
    }
}

} // namespace backend

} // namespace myslam
