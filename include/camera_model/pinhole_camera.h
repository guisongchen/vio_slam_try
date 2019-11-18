#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include "camera_model/camera.h"
#include <ceres/rotation.h>

namespace myslam
{

class PinholeCamera : public Camera
{
public:
    class Parameters : public Camera::Parameters
    {
    public:
        Parameters();
        bool readParamFromYamlFile(const std::string& filename);
        void copyParametersToArray(double* param) const;

    private:
        double k1_;
        double k2_;
        double p1_;
        double p2_;
        double fx_;
        double fy_;
        double cx_;
        double cy_;
    };

    PinholeCamera();

    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;
    const Parameters& getParameters() const;
    void setParameters(const Parameters& parameters);

private:
    Parameters parameters_;

    double  invK11_, invK13_, invK22_, invK23_;
    bool    noDistortionFlag_;
};

typedef std::shared_ptr<PinholeCamera> PinholeCameraPtr;
typedef std::shared_ptr<const PinholeCamera> PinholeCameraConstPtr;

}

#endif

