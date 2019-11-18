#include "camera_model/pinhole_camera.h"
#include <cmath>
#include <cstdio>
#include <opencv2/core/eigen.hpp>

namespace myslam
{

PinholeCamera::Parameters::Parameters()
    : Camera::Parameters(PINHOLE), 
      k1_ (0.0), k2_ (0.0), p1_ (0.0), p2_ (0.0),
      fx_ (0.0), fy_ (0.0), cx_ (0.0), cy_ (0.0) {}

bool PinholeCamera::Parameters::readParamFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("PINHOLE") != 0)
        {
            return false;
        }
    }

    modelType_ = PINHOLE;
    fs["camera_name"] >> cameraName_;
    imageWidth_ = static_cast<int>(fs["image_width"]);
    imageHeight_ = static_cast<int>(fs["image_height"]);

    cv::FileNode n = fs["distortion_parameters"];
    k1_ = static_cast<double>(n["k1"]);
    k2_ = static_cast<double>(n["k2"]);
    p1_ = static_cast<double>(n["p1"]);
    p2_ = static_cast<double>(n["p2"]);

    n = fs["projection_parameters"];
    fx_ = static_cast<double>(n["fx"]);
    fy_ = static_cast<double>(n["fy"]);
    cx_ = static_cast<double>(n["cx"]);
    cy_ = static_cast<double>(n["cy"]);
    
    std::cout << "PinholeCamera "
        << "\n  cameraName: " << cameraName_
        << "\n  imageWidth: " << imageWidth_
        << "\n  imageHeight: " << imageHeight_
        << "\n  k1: " << k1_ 
        << "\n  k2: " << k2_
        << "\n  p1: " << p1_
        << "\n  p2: " << p2_
        << "\n  fx: " << fx_
        << "\n  fy: " << fy_
        << "\n  cx: " << cx_
        << "\n  cy: " << cy_ << std::endl << std::endl;
    
    return true;
}

void PinholeCamera::Parameters::copyParametersToArray(double* param) const
{
    param[0] = k1_;
    param[1] = k2_;
    param[2] = p1_;
    param[3] = p2_;
    param[4] = fx_;
    param[5] = fy_;
    param[6] = cx_;
    param[7] = cy_;
}

PinholeCamera::PinholeCamera()
 : invK11_(1.0), invK13_(0.0), invK22_(1.0), invK23_(0.0), noDistortionFlag_(true) {}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p2d image coordinates
 * \param p3d coordinates of the projective ray
 */
void PinholeCamera::liftProjective(const Eigen::Vector2d& p2d, Eigen::Vector3d& p3d) const
{
    double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    //double lambda;

    // Lift points to normalised plane
    mx_d = invK11_ * p2d(0) + invK13_;
    my_d = invK22_ * p2d(1) + invK23_;

    if (noDistortionFlag_)
    {
        mx_u = mx_d;
        my_u = my_d;
    }
    
    // for fisheye camera, consider lens distortion
    else
    {
        // propose two ways to get lift 2d to 3d
        
        if (0) // just for debug, if choose inverse distortion model, set 0 to 1
        {
            double param[8];
            parameters_.copyParametersToArray(param);
            
            const double k1 = param[0];
            const double k2 = param[1];
            const double p1 = param[2];
            const double p2 = param[3];

            // Apply inverse distortion model
            // proposed by Heikkila
            mx2_d = mx_d*mx_d;
            my2_d = my_d*my_d;
            mxy_d = mx_d*my_d;
            
            rho2_d = mx2_d+my2_d;
            rho4_d = rho2_d*rho2_d;
            radDist_d = k1*rho2_d+k2*rho4_d;
            
            Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
            Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
            inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

            mx_u = mx_d - inv_denom_d*Dx_d;
            my_u = my_d - inv_denom_d*Dy_d;
        }
        else // another way, use recursive distortion model
        {
            // Recursive distortion model
            int n = 8;
            Eigen::Vector2d d_u;
            distortion(Eigen::Vector2d(mx_d, my_d), d_u);
            
            // Approximate value
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);

            for (int i = 1; i < n; ++i)
            {
                distortion(Eigen::Vector2d(mx_u, my_u), d_u);
                mx_u = mx_d - d_u(0);
                my_u = my_d - d_u(1);
            }
        }
    }

    // Obtain a projective ray
    p3d << mx_u, my_u, 1.0;
}



/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeCamera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const
{
    double param[8];
    parameters_.copyParametersToArray(param);
    
    const double k1 = param[0];
    const double k2 = param[1];
    const double p1 = param[2];
    const double p2 = param[3];

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

const PinholeCamera::Parameters& PinholeCamera::getParameters() const
{
    return parameters_;
}

void PinholeCamera::setParameters(const PinholeCamera::Parameters& parameters)
{
    parameters_ = parameters;
    
    double param[8];
    parameters_.copyParametersToArray(param);
    
    const double k1 = param[0];
    const double k2 = param[1];
    const double p1 = param[2];
    const double p2 = param[3];
    
    const double fx = param[4];
    const double fy = param[5];
    const double cx = param[6];
    const double cy = param[7];

    if (( k1 == 0.0) && ( k2 == 0.0) && ( p1 == 0.0) && ( p2 == 0.0))
    {
        noDistortionFlag_ = true;
    }
    else
    {
        noDistortionFlag_ = false;
    }

    invK11_ = 1.0 / fx;
    invK13_ = -cx / fx;
    invK22_ = 1.0 / fy;
    invK23_ = -cy / fy;
}

}
