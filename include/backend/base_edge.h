#ifndef MYSLAM_BACKEND_BASE_EDGE_H
#define MYSLAM_BACKEND_BASE_EDGE_H

#include "loss_function.h"
#include "base_vertex.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace myslam
{

namespace backend
{

extern unsigned long global_edge_id;

class Edge 
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Edge(int residual_dimension, int num_verticies);
    virtual ~Edge() {}

    bool AddVertex(std::shared_ptr<Vertex> vertex);
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices);
    
    std::shared_ptr<Vertex> GetVertex(int i) { return verticies_[i]; }
    std::vector<std::shared_ptr<Vertex>> Verticies() const { return verticies_; }

    size_t NumVertices() const { return verticies_.size(); }

    virtual void ComputeResidual() = 0;
    virtual void ComputeJacobians() = 0;

    double Chi2() const;
    double RobustChi2() const;

    Eigen::VectorXd Residual() const { return residual_; }
    std::vector<Eigen::MatrixXd> Jacobians() const { return jacobians_; }

    void SetInformation(const Eigen::MatrixXd &information);
    Eigen::MatrixXd Information() const { return information_; }
    Eigen::MatrixXd SqrtInformation() const { return sqrt_information_; }

    void SetLossFunction(LossFunction* ptr) { lossfunction_ = ptr; }
    LossFunction* GetLossFunction() { return lossfunction_;}
    
    void RobustInfo(double& drho, Eigen::MatrixXd& info) const;

    void SetObservation(const Eigen::VectorXd &observation) { observation_ = observation; }
    Eigen::VectorXd Observation() const { return observation_; }
    
    unsigned long Id() const { return id_; }
    int OrderingId() const { return ordering_id_; }
    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long                 id_; 
    int                           ordering_id_; 

    std::vector<std::shared_ptr<Vertex> > verticies_; 
    Eigen::VectorXd               residual_; 
    std::vector<Eigen::MatrixXd>  jacobians_; 
    Eigen::MatrixXd               information_;   
    Eigen::MatrixXd               sqrt_information_;
    Eigen::VectorXd               observation_;

    LossFunction*                 lossfunction_;
};

} // namespace backend
} // namespace myslam

#endif
