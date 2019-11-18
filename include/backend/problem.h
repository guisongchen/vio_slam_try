#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "base_edge.h"
#include "base_vertex.h"
#include "algorithm_LM.h"

namespace myslam 
{

namespace backend
{

typedef std::map<unsigned long, std::shared_ptr<Vertex>> MapVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Problem();
    ~Problem() {}

    bool AddVertex(std::shared_ptr<Vertex> vertex);
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);
    
    bool AddEdge(std::shared_ptr<Edge> edge);
    bool RemoveEdge(std::shared_ptr<Edge> edge);
    
    AlgorithmLM* Algorithm() const { return algorithmLM_; }
    void SetAlgorithm(AlgorithmLM* algorithmLM) { algorithmLM_ = algorithmLM; }
    
    bool SolveProblem(const int iterations = 10);
    
    double InitRobustChi2();
    double UpdateRobustChi2();
    
    double MaxHessianDiagonal();
    void SetOrdering();
    void MakeHessian();
    void SolveLinearSystem();
    void UpdateStates();
    void RollbackStates();
    
    bool ReportFlag() const { return reportFlag_; }
    void SetReport(bool reportFlag = false) { reportFlag_ = reportFlag; }
    
    Eigen::VectorXd DeltaX() const{ return delta_x_; }    
    Eigen::VectorXd ErrorVector() const { return b_; }
    double FinalCost() const { return finalCost_; }

    bool Marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex, int all_poses_dim);
    void AddPriorInfo(const Eigen::MatrixXd& H_prior, const Eigen::VectorXd& b_prior);
    void ResizeHessianPrior(const int dim);
    Eigen::MatrixXd GetHessianPrior() { return H_prior_; }
    Eigen::VectorXd GetbPrior() { return b_prior_; }
    
    int verticeSize() const  { return verticies_.size(); }

private:

    HashEdge GetConnectedEdges(std::shared_ptr<Vertex> vertex);
    void ComputeHessianBlock(HashEdge& edges, Eigen::MatrixXd& H, Eigen::VectorXd& b);
    
    AlgorithmLM*    algorithmLM_;
    
    MapVertex       verticies_;
    MapVertex       idx_pose_vertices_;
    MapVertex       idx_landmark_vertices_;
    MapVertex       verticies_marg_;

    HashEdge        edges_;
    HashVertexIdToEdge vertexToEdge_;
    
    Eigen::MatrixXd Hessian_;
    Eigen::VectorXd b_;
    Eigen::VectorXd delta_x_;

    Eigen::MatrixXd H_prior_;
    Eigen::VectorXd b_prior_;
    Eigen::VectorXd b_prior_backup_;
    Eigen::VectorXd err_prior_backup_;

    Eigen::MatrixXd Jt_prior_inv_;
    Eigen::VectorXd err_prior_;

    Eigen::MatrixXd H_pp_schur_;
    Eigen::VectorXd b_pp_schur_;

    Eigen::MatrixXd H_pp_;
    Eigen::VectorXd b_pp_;
    Eigen::MatrixXd H_ll_;
    Eigen::VectorXd b_ll_;

    unsigned long   ordering_poses_;
    unsigned long   ordering_landmarks_;
    unsigned long   ordering_total_;
        
    double          finalCost_;
    
    bool            priorFlag_;
    bool            reportFlag_;
}; 

    
} // namespace backend
    
} // namespace myslam

#endif
