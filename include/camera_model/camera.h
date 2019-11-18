#ifndef CAMERA_H
#define CAMERA_H

#include "common_include.h"

namespace myslam
{

class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum ModelType
    {
        KANNALA_BRANDT,
        MEI,
        PINHOLE
    };

    class Parameters
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        Parameters(ModelType modelType);
        virtual bool readParamFromYamlFile(const std::string& filename) = 0;
        
    protected:
        ModelType   modelType_;
        int         intrinsicNum_;
        std::string cameraName_;
        int         imageWidth_;
        int         imageHeight_;
    };

    virtual void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const = 0;

protected:
    cv::Mat mask_;
};

typedef std::shared_ptr<Camera> CameraPtr;
typedef std::shared_ptr<const Camera> CameraConstPtr;

}

#endif
