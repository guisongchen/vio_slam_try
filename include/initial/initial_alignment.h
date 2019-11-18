#ifndef INITIAL_ALIGNMENT_H
#define INITIAL_ALIGNMENT_H

#include "common_include.h"
#include "backend/integration_base.h"

namespace myslam
{
    
class ImageFrame
{
  
public:
    ImageFrame() {};
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > &image,
               double header) : header_ {header}, isKeyframe_ {false}
    {
        image_ = image;
    };
    
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > image_;
    double      header_;
    Matrix3d    rotation_;
    Vector3d    translation_;
    IntegrationBase* preIntegration_;
    bool        isKeyframe_;
    
};
    
bool VisualIMUAlignment(std::map<double, ImageFrame> &allImageFrame,
                        Vector3d *Bgs, Vector3d &g, VectorXd &x);

}

#endif
