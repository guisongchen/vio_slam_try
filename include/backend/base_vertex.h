#ifndef MYSLAM_BACKEND_BASE_VERTEX_H
#define MYSLAM_BACKEND_BASE_VERTEX_H

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace myslam 
{
namespace backend
{
extern unsigned long global_vertex_id;

class Vertex
{

public:
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    enum VertexType
    {
        POSE,
        LANDMARK
    };

    explicit Vertex(int num_dimension, int local_dimension = -1);
    virtual ~Vertex() {}

    int Dimension() const { return param_dimension_; }
    
    int LocalDimension() const { return local_dimension_; }

    unsigned long Id() const { return id_;}
    
    int OrderingId() const { return ordering_id_; }
    
    void SetOrderingId(unsigned long id) 
    { 
        ordering_id_ = id; 
    }

    Eigen::VectorXd Parameters() const 
    {
        return parameters_;
    }
    
    Eigen::VectorXd &Parameters()
    {
        return parameters_;
    }
    
    void SetParameters(const Eigen::VectorXd &params) { parameters_ = params; }
    
    void BackUpParameters() 
    {
        parameters_backup_ = parameters_; 
    }
    
    void RollBackParameters() 
    {
        parameters_ = parameters_backup_; 
    }

    void setVertexType(VertexType vertex_type) 
    { 
        vertex_type_ = vertex_type;
    }
    
    VertexType TypeInfo() const 
    {
        return vertex_type_;
    }

    void SetFixed(bool fixed = true) 
    {
        fixed_ = fixed;
    }
    
    bool IsFixed() const 
    {
        return fixed_;
    }
    
    virtual void Plus(const Eigen::VectorXd &delta);

protected:
    
    VertexType                 vertex_type_;
    Eigen::VectorXd            parameters_;
    Eigen::VectorXd            parameters_backup_;
    int                        param_dimension_;
    int                        local_dimension_;
    unsigned long              id_;
    
    // column starting idx in Hessian matrix, range: (ordering_id_, ordering_id_ + local_dimension)
    unsigned long              ordering_id_;
    bool                       fixed_;

};

} // namespace backend
} // namespace myslam

#endif

