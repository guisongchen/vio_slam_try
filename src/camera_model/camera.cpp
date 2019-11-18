#include "camera_model/camera.h"
#include <boost/algorithm/string.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace myslam
{

Camera::Parameters::Parameters(ModelType modelType)
 : modelType_(modelType), imageWidth_(0), imageHeight_(0)
{
    switch (modelType)
    {
        case KANNALA_BRANDT:
            intrinsicNum_ = 8;
            break;
        case PINHOLE:
            intrinsicNum_ = 8;
            break;
        case MEI:
        default:
            intrinsicNum_ = 9;
    }
}

}
