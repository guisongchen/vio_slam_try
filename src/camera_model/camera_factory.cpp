#include "camera_model/camera_factory.h"
#include "camera_model/pinhole_camera.h"
#include <boost/algorithm/string.hpp>

#include "ceres/ceres.h"

namespace myslam
{

std::shared_ptr<CameraFactory> CameraFactory::cameraInstance_;

CameraFactory::CameraFactory() {}

std::shared_ptr<CameraFactory> CameraFactory::generateCameraInstance(void)
{
    if ( cameraInstance_.get() == 0)
        cameraInstance_.reset(new CameraFactory);

    return cameraInstance_;
}

CameraPtr CameraFactory::generateCameraFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return CameraPtr();
    }

    Camera::ModelType modelType = Camera::MEI;
    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (boost::iequals(sModelType, "pinhole"))
        {
            modelType = Camera::PINHOLE;
        }
        else
        {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
            return CameraPtr();
        }
    }

    switch (modelType)
    {
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera(new PinholeCamera);

            PinholeCamera::Parameters params = camera->getParameters();
            params.readParamFromYamlFile(filename);
            camera->setParameters(params);
            return camera;
        }
        default:
            return CameraPtr();
    }

    return CameraPtr();
}

}


