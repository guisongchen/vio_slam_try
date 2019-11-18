#ifndef CAMERAFACTORY_H
#define CAMERAFACTORY_H

#include "camera_model/camera.h"

namespace myslam
{

class CameraFactory
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraFactory();

    static std::shared_ptr<CameraFactory> generateCameraInstance(void);

    CameraPtr generateCameraFromYamlFile(const std::string& filename);

private:
    static std::shared_ptr<CameraFactory> cameraInstance_;
};

}

#endif
