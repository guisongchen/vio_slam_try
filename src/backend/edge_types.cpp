#include "backend/edge_types.h"

namespace myslam
{
namespace backend
{
    
//---------------IMU------------------------------------

void EdgeImu::ComputeResidual()
{
    Eigen::VectorXd param_0 = verticies_[0]->Parameters();
    Eigen::Quaterniond Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Eigen::Vector3d Pi = param_0.head<3>();
    

    Eigen::VectorXd param_1 = verticies_[1]->Parameters();
    Eigen::Vector3d Vi = param_1.head<3>();
    Eigen::Vector3d Bai = param_1.segment(3, 3);
    Eigen::Vector3d Bgi = param_1.tail<3>();

    Eigen::VectorXd param_2 = verticies_[2]->Parameters();
    Eigen::Quaterniond Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Eigen::Vector3d Pj = param_2.head<3>();

    Eigen::VectorXd param_3 = verticies_[3]->Parameters();
    Eigen::Vector3d Vj = param_3.head<3>();
    Eigen::Vector3d Baj = param_3.segment(3, 3);
    Eigen::Vector3d Bgj = param_3.tail<3>();

    residual_ = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                           Pj, Qj, Vj, Baj, Bgj);
    
    SetInformation(pre_integration_->covariance.inverse());
}


// we handle infomation effect on jacobians in solving linear problem function
void EdgeImu::ComputeJacobians()
{

    Eigen::VectorXd param_0 = verticies_[0]->Parameters();
    Eigen::Quaterniond Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Eigen::Vector3d Pi = param_0.head<3>();

    Eigen::VectorXd param_1 = verticies_[1]->Parameters();
    Eigen::Vector3d Vi = param_1.head<3>();
//     Eigen::Vector3d Bai = param_1.segment(3, 3);
    Eigen::Vector3d Bgi = param_1.tail<3>();

    Eigen::VectorXd param_2 = verticies_[2]->Parameters();
    Eigen::Quaterniond Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Eigen::Vector3d Pj = param_2.head<3>();

    Eigen::VectorXd param_3 = verticies_[3]->Parameters();
    Eigen::Vector3d Vj = param_3.head<3>();
//     Eigen::Vector3d Baj = param_3.segment(3, 3);
//     Eigen::Vector3d Bgj = param_3.tail<3>();

    double sum_dt = pre_integration_->sum_dt;
    
    Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);

    if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8)
        cout << "numerical unstable in preintegration" << endl;
    
    //-------pose i--------------
    {

        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
        jacobian_pose_i.setZero();
        
        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * 
                                                (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * 
                                               Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
                                            
        jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) *
                                                  Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
        
        jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
            cout << "numerical unstable in preintegration" << endl;
        
        jacobians_[0] = jacobian_pose_i;
    }
    
    //-------speedbias i----------
    {

        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
        jacobian_speedbias_i.setZero();
        
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q *
                                               Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
                                            
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi *
                                                            corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                                                            
//         jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi *
//                                                     pre_integration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                                                            
        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

        jacobians_[1] = jacobian_speedbias_i;
    }
    
    //-------pose j----------
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
        jacobian_pose_j.setZero();

        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * 
                                               Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
                                               
        jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() *
                                                Qi.inverse() * Qj).bottomRightCorner<3, 3>();
                                                
        jacobians_[2] = jacobian_pose_j;

    }
    
    //-------speedbias j----------
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
        jacobian_speedbias_j.setZero();

        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

        jacobians_[3] = jacobian_speedbias_j;

    }
}

//--------------------invdepth reprojection-----------------------

void EdgeReprojection::ComputeResidual()
{
    double inv_dep_i = verticies_[0]->Parameters()[0];

    Eigen::VectorXd param_i = verticies_[1]->Parameters();
    Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Eigen::Vector3d Pi = param_i.head<3>();

    Eigen::VectorXd param_j = verticies_[2]->Parameters();
    Eigen::Quaterniond Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Eigen::Vector3d Pj = param_j.head<3>();

    Eigen::VectorXd param_ext = verticies_[3]->Parameters();
    Eigen::Quaterniond qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Eigen::Vector3d tic = param_ext.head<3>();

    Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();
    residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
}

// TODO resize jacobian matrix when add vertex
void EdgeReprojection::ComputeJacobians()
{
    double inv_dep_i = verticies_[0]->Parameters()[0];

    Eigen::VectorXd param_i = verticies_[1]->Parameters();
    Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Eigen::Vector3d Pi = param_i.head<3>();

    Eigen::VectorXd param_j = verticies_[2]->Parameters();
    Eigen::Quaterniond Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Eigen::Vector3d Pj = param_j.head<3>();

    Eigen::VectorXd param_ext = verticies_[3]->Parameters();
    Eigen::Quaterniond qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Eigen::Vector3d tic = param_ext.head<3>();

    Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();

    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    reduce << 1. / dep_j,   0,            -pts_camera_j(0) / (dep_j * dep_j),
              0,            1. / dep_j,   -pts_camera_j(1) / (dep_j * dep_j);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

    Eigen::Vector2d jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
    Eigen::Matrix<double, 3, 6> jaco_ex;
    jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
    Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
    jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                             Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
    jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;
    jacobians_[2] = jacobian_pose_j;
    jacobians_[3] = jacobian_ex_pose;
}

//--------------------XYZ reprojection-----------------------

void EdgeReprojectXYZ::ComputeResidual()
{
    Eigen::VectorXd pose = verticies_[0]->Parameters();
    Eigen::Quaterniond quat(pose[6], pose[3], pose[4], pose[5]);
    Eigen::Vector3d trans = pose.head<3>();
    Eigen::Vector3d point = verticies_[1]->Parameters();
    
    residual_ = measurement_ - Camera2Pixel(quat * point + trans);
}

void EdgeReprojectXYZ::ComputeJacobians()
{
    Eigen::VectorXd pose = verticies_[0]->Parameters();
    Eigen::Quaterniond quat(pose[6], pose[3], pose[4], pose[5]);
    Eigen::Vector3d trans = pose.head<3>();
    
    Eigen::Vector3d point = verticies_[1]->Parameters();
    Eigen::Vector3d point_trans = quat * point + trans;
    
    const double x = point_trans[0];
    const double y = point_trans[1];
    const double z = point_trans[2];
    const double z_2 = z * z;
    
    Eigen::Matrix<double, 2, 3> jacobian_uv_xyzTrans;
    jacobian_uv_xyzTrans(0, 0) = fx_;
    jacobian_uv_xyzTrans(0, 1) = 0;
    jacobian_uv_xyzTrans(0, 2) = -x / z * fx_;

    jacobian_uv_xyzTrans(1, 0) = 0;
    jacobian_uv_xyzTrans(1, 1) = fy_;
    jacobian_uv_xyzTrans(1, 2) = -y / z * fy_;
    
    Eigen::Matrix<double, 2, 6> jacobians_uv_kesai;
    jacobians_uv_kesai(0, 0) = x * y / z_2 * fx_;
    jacobians_uv_kesai(0, 1) = -(1 + (x * x / z_2)) * fx_;
    jacobians_uv_kesai(0, 2) = y / z * fx_;
    jacobians_uv_kesai(0, 3) = -1. / z * fx_;
    jacobians_uv_kesai(0, 4) = 0;
    jacobians_uv_kesai(0, 5) = x / z_2 * fx_;

    jacobians_uv_kesai(1, 0) = (1 + y * y / z_2) * fy_;
    jacobians_uv_kesai(1, 1) = -x * y / z_2 * fy_;
    jacobians_uv_kesai(1, 2) = -x / z * fy_;
    jacobians_uv_kesai(1, 3) = 0;
    jacobians_uv_kesai(1, 4) = -1. / z * fy_;
    jacobians_uv_kesai(1, 5) = y / z_2 * fy_;
    
    jacobians_[0] = jacobians_uv_kesai;
    jacobians_[1] = -1.0 / z * jacobian_uv_xyzTrans * quat.toRotationMatrix();
}

void EdgeReprojectXYZ::SetCameraIntrisic(const double fx, const double fy, const double cx, const double cy)
{
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
}

Eigen::Vector2d EdgeReprojectXYZ::Camera2Pixel(const Eigen::Vector3d& point) const
{
    Eigen::Vector2d p2d(point[0]/point[2], point[1]/point[2]);
    const double u = fx_ * p2d[0] + cx_; 
    const double v = fy_ * p2d[1] + cy_;
    
    return Eigen::Vector2d(u, v);
}



} // namespace backend
} // namespace myslam

