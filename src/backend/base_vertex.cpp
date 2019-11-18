#include "backend/base_vertex.h"

namespace myslam
{
namespace backend
{

unsigned long global_vertex_id;
    
Vertex::Vertex(int num_dimension, int local_dimension) : param_dimension_(num_dimension)
{
    parameters_.resize(param_dimension_, 1);
    local_dimension_ = local_dimension > 0 ? local_dimension : param_dimension_;
    
    id_ = global_vertex_id++;
    ordering_id_ = 0;
    fixed_ = false;
}

void Vertex::Plus(const Eigen::VectorXd &delta) { parameters_ += delta; }
    
} // namespace backend

} // namespace myslam
