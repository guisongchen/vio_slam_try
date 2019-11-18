#include "backend/problem.h"

#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <cmath>

namespace myslam
{

namespace backend
{

Problem::Problem()
    : algorithmLM_(nullptr), ordering_poses_(0), ordering_landmarks_(0), ordering_total_(0),
      finalCost_(-1.0), priorFlag_(false), reportFlag_(false)
{
    verticies_marg_.clear();
    global_vertex_id = 0;
    global_edge_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex)
{
    if (verticies_.find(vertex->Id()) != verticies_.end())
    {
        std::cerr << "vertex " << vertex->Id() << " already exist!" << std::endl;
        return false;
    }
    else
        verticies_.insert(std::make_pair(vertex->Id(), vertex));
    
    if (priorFlag_ && vertex->TypeInfo() == Vertex::POSE)
        ResizeHessianPrior(vertex->LocalDimension());
    
    return true;
}

    //  H * x = b
    //
    //  | x x 0 0 0 |           | xm |      | bm |
    //  | x x 0 0 0 |_ _        | xm |      | bm |
    //  | 0 0 0 0 0 | |         | xr |      | 0  |
    //  | 0 0 0 0 0 |dim        | xr |      | 0  |
    //  | 0 0 0 0 0 |_|_        | xr |      | 0  |
    //  -----|-dim--|
    // 

void Problem::ResizeHessianPrior(const int dim)
{
    int size = H_prior_.rows() + dim;
    
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);
    
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
    b_prior_.tail(dim).setZero();
}


bool Problem::AddEdge(std::shared_ptr<Edge> edge)
{
    if (edges_.find(edge->Id()) == edges_.end())
        edges_.insert(std::make_pair(edge->Id(), edge));
    else
    {
        std::cerr << "Edge " << edge->Id() << " already exist!" << std::endl;
        return false;
    }

    for (auto &vertex: edge->Verticies())
        vertexToEdge_.insert(std::make_pair(vertex->Id(), edge));

    return true;
}

void Problem::AddPriorInfo(const Eigen::MatrixXd& H_prior, const Eigen::VectorXd& b_prior)
{
    H_prior_ = H_prior;
    b_prior_ = b_prior;
    
    const double eps = 1e-8;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;
    
    ResizeHessianPrior(15);
    
    priorFlag_ = true;
    
}

HashEdge Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex)
{
    HashEdge edges;
    
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter)
    {
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.insert(std::make_pair(iter->second->Id(), iter->second));
    }
    
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex)
{
    if (verticies_.find(vertex->Id()) == verticies_.end())
    {
        std::cerr << "The vertex " << vertex->Id() << " is not in the problem!" << std::endl;
        return false;
    }

    HashEdge remove_edges = GetConnectedEdges(vertex);
    for (auto edge : remove_edges)
        RemoveEdge(edge.second);

    if (vertex->TypeInfo() == Vertex::POSE)
        idx_pose_vertices_.erase(vertex->Id());
    else if (vertex->TypeInfo() == Vertex::LANDMARK)
        idx_landmark_vertices_.erase(vertex->Id());
    
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}


bool Problem::RemoveEdge(std::shared_ptr<Edge> edge)
{
    if (edges_.find(edge->Id()) == edges_.end())
    {
        std::cerr << "The edge " << edge->Id() << " is not in the problem!" << std::endl;
        return false;
    }
    edges_.erase(edge->Id());
    
    return true;
}

bool Problem::SolveProblem(const int iterations)
{
    if (edges_.size() == 0 || verticies_.size() == 0)
    {
        std::cerr << "graph is empty, no vertexs or edges" << std::endl;
        return false;
    }
    
    if (!algorithmLM_)
    {
        std::cerr << "algorithm is null, maybe forget to call SetAlgorithm() fuction.." << std::endl;
        return false;
    }
    
    SetOrdering();
    MakeHessian();
    
    if (reportFlag_)
    {
        std::cout << "----graph info-------------" << "\n"
                  << "verticies = " << verticies_.size() 
                  << " (pose: " << idx_pose_vertices_.size()
                  << "; landmarks: " << idx_landmark_vertices_.size()
                  << ") edges = " << edges_.size() << std::endl;
    }
    
    bool result = algorithmLM_->Optimize(iterations);
    
    finalCost_ = algorithmLM_->CurrentChi2();
    
    return result;
}

void Problem::SetOrdering()
{
    ordering_poses_ = 0;
    ordering_total_ = 0;
    ordering_landmarks_ = 0;

    for (auto vertexInfo: verticies_)
    {
        // sum of all verticies local dimensions
        ordering_total_ += vertexInfo.second->LocalDimension();
        
        if (vertexInfo.second->TypeInfo() == Vertex::POSE)
        {
            vertexInfo.second->SetOrderingId(ordering_poses_);
            idx_pose_vertices_.insert(std::make_pair(ordering_poses_, vertexInfo.second));
            ordering_poses_ += vertexInfo.second->LocalDimension();
        }
        else if (vertexInfo.second->TypeInfo() == Vertex::LANDMARK)
        {
            vertexInfo.second->SetOrderingId(ordering_landmarks_);
            idx_landmark_vertices_.insert(std::make_pair(ordering_landmarks_, vertexInfo.second));
            ordering_landmarks_ += vertexInfo.second->LocalDimension();
        }
        
    }

    // update landmark ordering id. set all pose ordering id at front, then comes landmarks.
    for (auto landmarkVertex : idx_landmark_vertices_)
        landmarkVertex.second->SetOrderingId(landmarkVertex.second->OrderingId() + ordering_poses_);
}

void Problem::MakeHessian()
{
    unsigned long size = ordering_total_;
    Eigen::MatrixXd H(size, size);
    Eigen::VectorXd b(size);
    H.setZero();
    b.setZero();
    
    ComputeHessianBlock(edges_, H, b);
    
    Hessian_ = H;
    b_ = b;


    if(priorFlag_)
    {
        Eigen::MatrixXd H_prior_tmp = H_prior_;
        Eigen::VectorXd b_prior_tmp = b_prior_;
        
        const int H_prior_cols = H_prior_tmp.cols();
        const int H_prior_rows = H_prior_tmp.rows();
        
        for (auto vertex: verticies_)
        {
            std::shared_ptr<Vertex> currVertex = vertex.second;
            
            if (currVertex->TypeInfo() == Vertex::POSE && currVertex->IsFixed())
            {
                // OrderingId is starting column-index in Hessian
                int idx = currVertex->OrderingId();
                int dim = currVertex->LocalDimension();
                
                //  H * x = b (this is just the pose part of whole Hessian)
                //                    ___
                //  | x x 0 0 0 |      |            | pose     |      | bm |
                //  | x x 0 0 0 |_ _   |            | pose     |      | bm |
                //  | 0 0 0 0 0 | |   ordering      | pose_fix |      | 0  |
                //  | 0 0 0 0 0 |dim  pose          | pose_fix |      | 0  |
                //  | 0 0 0 0 0 |_|_  _|_           | pose_fix |      | 0  |
                //       |-dim- |     
                // 
                // since vertex is fixed, J = 0, J^T * J area should be zero, so does residual
                
                H_prior_tmp.block(idx, 0, dim, H_prior_cols).setZero();
                H_prior_tmp.block(0, idx, H_prior_rows, dim).setZero();
                b_prior_tmp.segment(idx, dim).setZero();
            }
        }
        
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    // initial delta_x = 0
    delta_x_ = Eigen::VectorXd::Zero(size);
}

void Problem::ComputeHessianBlock(HashEdge& edges, Eigen::MatrixXd& H, Eigen::VectorXd& b)
{
    for (auto &edge: edges)
    {
        std::shared_ptr<Edge> currEdge = edge.second;
        
        currEdge->ComputeResidual();
        currEdge->ComputeJacobians();
        auto jacobians = currEdge->Jacobians();
        auto verticies = currEdge->Verticies();
        
        assert(jacobians.size() == verticies.size());
        
        double drho; // aka rho[1]
        const int infoSize = currEdge->Information().rows();
        Eigen::MatrixXd robustInfo(infoSize, infoSize);
        currEdge->RobustInfo(drho,robustInfo);
        
        for (size_t i = 0; i < verticies.size(); ++i)
        {
            auto v_i = verticies[i];
            if (v_i->IsFixed())
                continue;    // fixed, which means value in H is 0

            // H = J^T * robust_info * J
            auto jacobian_i = jacobians[i];
            Eigen::MatrixXd JtW = jacobian_i.transpose() * robustInfo;
            
            unsigned long index_i = v_i->OrderingId();
            unsigned long dim_i = v_i->LocalDimension();
            
            for (size_t j = i; j < verticies.size(); ++j)
            {
                auto v_j = verticies[j];
                if (v_j->IsFixed())
                    continue;

                auto jacobian_j = jacobians[j];
                unsigned long index_j = v_j->OrderingId();
                unsigned long dim_j = v_j->LocalDimension();
                
                Eigen::MatrixXd hessian = JtW * jacobian_j;

                // add sub block together, use noalias avoiding performance cost
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                
                if (j != i) 
                {
                    // H is symmetry matrix
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
            }
            
            // b = -rho[1] * J^T * info * residual
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * 
                                                   currEdge->Information() * currEdge->Residual();
        }
    }
    
}


double Problem::MaxHessianDiagonal()
{
    double maxDiagonal = 0;
    unsigned long size = Hessian_.cols();
    
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    
    for (unsigned long i = 0; i < size; ++i)
        maxDiagonal = std::max(std::fabs(Hessian_(i, i)), maxDiagonal);
    
    maxDiagonal = std::min(1e10, maxDiagonal);
    
    return maxDiagonal;
}

double Problem::InitRobustChi2()
{
    double currentChi2 = 0;
    
    // residual has computed when make hessian
    for (auto edge: edges_)
        currentChi2 += edge.second->RobustChi2();
    
    if (priorFlag_)
        currentChi2 += err_prior_.squaredNorm();
    
    return 0.5 * currentChi2;
}

double Problem::UpdateRobustChi2()
{
    double currentChi2 = 0;
    
    for (auto edge: edges_)
    {
        edge.second->ComputeResidual();
        currentChi2 += edge.second->RobustChi2();
    }
    
    if (priorFlag_)
        currentChi2 += err_prior_.squaredNorm();
    
    return 0.5 * currentChi2;
}


void Problem::SolveLinearSystem()
{
    // step1: schur marginalization --> Hpp, bpp
    int reserve_size = ordering_poses_;
    int marg_size = ordering_landmarks_;
    
    // | Hpp | Hpm |  <--pose (reserve)             | bp | 
    // |-----|-----|                                |----|
    // | Hmp | Hmm |  <--landmark (margialize)      | bm |
    
    Eigen::MatrixXd Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
    Eigen::MatrixXd Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
    Eigen::MatrixXd Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
    Eigen::VectorXd bpp = b_.segment(0, reserve_size);
    Eigen::VectorXd bmm = b_.segment(reserve_size, marg_size);
    
    Eigen::MatrixXd Hmm_inv(Eigen::MatrixXd::Zero(marg_size, marg_size));

    for (auto landmarkVertex : idx_landmark_vertices_)
    {
        // inverse Hessian block for each landmark
        int idx = landmarkVertex.second->OrderingId() - reserve_size;
        int size = landmarkVertex.second->LocalDimension();
        
        Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
    }
    
    // | Hpp - Hpm * Hmm_inv * Hmp | 0   |  <--pose (reserve)        | xp |     | bp - Hpm * Hmm_inv * bm | 
    // |-------------------------- |-----|                           |----|     |-------------------------|
    // | Hmp                       | Hmm |  <--landmark (margialize) | xm |     | bm                      |

    Eigen::MatrixXd tempH = Hpm * Hmm_inv;
    H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
    b_pp_schur_ = bpp - tempH * bmm;

    // step2: solve H_pp_schur_ * delta_x_pp = b_pp_schur_
    Eigen::VectorXd delta_x_pp(Eigen::VectorXd::Zero(reserve_size));

    // (H + lamda*I) * delta_x = b
    double currentLambda = algorithmLM_->CurrentLambda();
    for (ulong i = 0; i < ordering_poses_; ++i)
        H_pp_schur_(i, i) += currentLambda;              // LM Method

    // cholesky decompose
    delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);  //  SVec.asDiagonal() * svd.matrixV() * Ub;    
    
    // fill in delta_x with pose optimzer result
    delta_x_.head(reserve_size) = delta_x_pp;

    // step3: solve  Hmp * delta_x_pp + Hmm * delta_xmm = bm
    Eigen::VectorXd delta_x_mm(marg_size);
    delta_x_mm = Hmm_inv * (bmm - Hmp * delta_x_pp);
    
    // fill in delta_x with landmark optimzer result
    delta_x_.tail(marg_size) = delta_x_mm;

}


void Problem::UpdateStates()
{
    // update vertex
    for (auto vertex: verticies_)
    {
        // backup last optimized value in case rollback
        vertex.second->BackUpParameters();

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        
        Eigen::VectorXd delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // if we need to consider prior information
    if (priorFlag_)
    {
        // backup prior in case rollback
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);
    }

}

void Problem::RollbackStates()
{
    // update vertex
    for (auto vertex: verticies_)
        vertex.second->RollBackParameters();

    // Roll back prior_
    if (priorFlag_)
    {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}


bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int all_poses_dim)
{
    
    SetOrdering();

    HashEdge marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, std::shared_ptr<Vertex> > margLandmark;
    
    // re-order landmark in Hessian, all_pose_dimension at front then comes landmarks
    int marg_landmark_size = 0;
    for (auto edge : marg_edges)
    {
        auto verticies = edge.second->Verticies();
        for (auto iter : verticies)
        {
            if (iter->TypeInfo() == Vertex::LANDMARK && margLandmark.find(iter->Id()) == margLandmark.end())
            {
                iter->SetOrderingId(all_poses_dim + marg_landmark_size);
                margLandmark.insert(std::make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }

    
    const int total_size = all_poses_dim + marg_landmark_size;
    Eigen::MatrixXd H_marg(Eigen::MatrixXd::Zero(total_size, total_size));
    Eigen::VectorXd b_marg(Eigen::VectorXd::Zero(total_size));
    
    ComputeHessianBlock(marg_edges, H_marg, b_marg);

    //
    // | Hpp | Hpm |  <--pose (reserve)             | bp | 
    // |-----|-----|                                |----|
    // | Hmp | Hmm |  <--landmark (margialize)      | bm |
    //
    //
    // | Hpp - Hpm * Hmm_inv * Hmp | 0   |  <--pose (reserve)        | xp |     | bp - Hpm * Hmm_inv * bm | 
    // |-------------------------- |-----|                           |----|     |-------------------------|
    // | Hmp                       | Hmm |  <--landmark (margialize) | xm |     | bm                      |
    
    
    // marg landmark
    const int reserve_size = all_poses_dim;
    if (marg_landmark_size > 0)
    {
        const int& marg_size = marg_landmark_size;
        Eigen::MatrixXd Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        Eigen::MatrixXd Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        Eigen::MatrixXd Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        Eigen::VectorXd bpp = b_marg.segment(0, reserve_size);
        Eigen::VectorXd bmm = b_marg.segment(reserve_size, marg_size);

        Eigen::MatrixXd Hmm_inv(Eigen::MatrixXd::Zero(marg_size, marg_size));

        for (auto iter: margLandmark)
        {
            int idx = iter.second->OrderingId() - reserve_size;
            int size = iter.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        Eigen::MatrixXd tempH = Hpm * Hmm_inv;
        Eigen::MatrixXd Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        bpp = bpp - tempH * bmm;

        H_marg = Hpp;
        b_marg = bpp;
    }

    Eigen::VectorXd b_prior_before = b_prior_;
    
    // update prior with last prior info
    if(priorFlag_)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }
    
    int marg_dim = 0;

    // starting from the last added one(aka speedbias of marg frame), then marg frame pose
    
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {

        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();

        marg_dim += dim;
        
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;
     
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }
    
    // following code is similar to VINS-Mono

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;
    
    // make sure Amm symmetric
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();
                              
    // | Hrr | Hrm |  <--pose (reserve, n2)         | br | 
    // |-----|-----|                                |----|
    // | Hmr | Hmm |  <--pose (margialize, m2)      | bm |
    //
    //
    // | Hpp - Hpm * Hmm_inv * Hmp | 0   |  <--pose (reserve)        | xp |     | bp - Hpm * Hmm_inv * bm | 
    // |-------------------------- |-----|                           |----|     |-------------------------|
    // | Hmp                       | Hmm |  <--landmark (margialize) | xm |     | bm                      |

    Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
    Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    Eigen::MatrixXd tempB = Arm * Amm_inv;

    H_prior_ = Arr - tempB * Amr;
    b_prior_ = brr - tempB * bmm2;


    for (size_t k = 0; k < margVertexs.size(); ++k)
        RemoveVertex(margVertexs[k]);

    for (auto landmarkVertex: margLandmark)
        RemoveVertex(landmarkVertex.second);

    return true;

}

    
} // namespace backend
    
} // namespace myslam
