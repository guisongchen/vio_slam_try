#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include "parameter.h"
#include <list>

namespace myslam
{

class CovisibleFeature
{
public:
    CovisibleFeature(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        
        uv.x() = _point(3);
        uv.y() = _point(4);
        
        velocity.x() = _point(5);
        velocity.y() = _point(6);

        cur_td = td;
    }

    double    cur_td;
    Vector3d  point;
    Vector2d  uv;
    Vector2d  velocity;
    
    double    z;
    bool      is_used;
    double    parallax;
    MatrixXd  A;
    VectorXd  b;
    double    dep_gradient;
};

class Feature
{
public:
    Feature(int featureId, int startFrameId)
        : featureId_(featureId), startFrameId_(startFrameId),
          covisibleNum_(0), estimatedDepth_(-1.0), solveFlag_(0) {}

    const int featureId_;
    int       startFrameId_;
    std::vector<CovisibleFeature> covisibleFeatures_;

    int       covisibleNum_;
    bool      is_outlier;
    bool      is_margin;
    double    estimatedDepth_;
    int       solveFlag_;   // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d  gt_p;

};

class FeatureManager
{
public:
    FeatureManager(Matrix3d Rs[]);

    void setRic(Matrix3d ric);
    bool addFeatureCheckParallax(int frame_count, 
                                const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > &image, 
                                double td);
    std::vector<std::pair<Vector3d, Vector3d> > getCorresponding(int queryRefFrameId, int queryMatchFrameId);
    int getFeatureCount();
    VectorXd getInvDepthVector();
    void clearDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void triangulate(Vector3d Ps[], Vector3d &tic, Matrix3d &ric);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, 
                              Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeOld();
    void removeSecondNew();
    void removeOutlier();
    void removeFailures();
    void clearState();
    
    int       lastTrackNum_;
    std::list<Feature> trackedFeatures_;
  
private:
    double computeParallax(const Feature &trackedFeature, int frameId);
    const Matrix3d *Rs_;
    Matrix3d  ric_;
  
};

}

#endif
