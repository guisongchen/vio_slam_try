#ifndef MYSLAM_BACKEND_VERTEX__TYPES_H
#define MYSLAM_BACKEND_VERTEX__TYPES_H

#include "base_vertex.h"

namespace myslam
{
namespace backend
{

class VertexPose : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexPose() : Vertex(7, 6) { setVertexType(POSE); }
    
    void SetParameterData(double* param);

    virtual void Plus(const Eigen::VectorXd &delta) override;
};


class VertexSpeedBias : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSpeedBias() : Vertex(9) { setVertexType(POSE); }
    
    void SetParameterData(double* param);

};


class VertexInverseDepth : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexInverseDepth() : Vertex(1) { setVertexType(LANDMARK); }
    
    void SetParameterData(double* param);
};

class VertexPointXYZ : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexPointXYZ() : Vertex(3) { setVertexType(LANDMARK); }
    
    void SetParameterData(double* param);
};


} // namespace backend
} // namespace myslam

#endif

