#include "initial/initial_alignment.h"

namespace myslam
{
 
void solveGyroscopeBias(std::map<double, ImageFrame> &allImageFrame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i; // b_k
    std::map<double, ImageFrame>::iterator frame_j; // b_k+1
   
    // frame_j is one step ahead of frame_i, use next(frame_i) to get iterator, make sure not point out
    for (frame_i = allImageFrame.begin(); next(frame_i) != allImageFrame.end(); frame_i++)
    {
        // point to next without change frame_i
        frame_j = next(frame_i);
        
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        
        VectorXd tmp_b(3);
        tmp_b.setZero();
        
        Eigen::Quaterniond q_ij(frame_i->second.rotation_.transpose() * frame_j->second.rotation_);
        
        tmp_A = frame_j->second.preIntegration_->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.preIntegration_->delta_q.inverse() * q_ij).vec();
        
        // sum of all frames
        // sum(A) * bg = sum(b)
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    
    delta_bg = A.ldlt().solve(b);
    
    cout << "gyroscope bias initial calibration " << delta_bg.transpose() << endl;

    // considering bias will be accumulated
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = allImageFrame.begin(); next(frame_i) != allImageFrame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.preIntegration_->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    
    if(a == tmp)
        tmp << 1, 0, 0;
    
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    
    return bc;
}

void RefineGravity(std::map<double, ImageFrame> &allImageFrame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();

    int frameNum = allImageFrame.size();
    
    // g magnitude is fixed, so DOF is 2
    int stateVectorNum = frameNum * 3 + 2 + 1;

    MatrixXd A{stateVectorNum, stateVectorNum};
    A.setZero();
    VectorXd b{stateVectorNum};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    
    // iterate 4 times
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        
        int i = 0;
        for (frame_i = allImageFrame.begin(); next(frame_i) != allImageFrame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.preIntegration_->sum_dt;
            
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.rotation_.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            
            tmp_A.block<3, 1>(0, 8) = frame_i->second.rotation_.transpose() * 
                                     (frame_j->second.translation_ - frame_i->second.translation_) / 100.0;     
            
            tmp_b.block<3, 1>(0, 0) = frame_j->second.preIntegration_->delta_p + frame_i->second.rotation_.transpose() *
                                      frame_j->second.rotation_ * TIC[0] - TIC[0] - 
                                      frame_i->second.rotation_.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.rotation_.transpose() * frame_j->second.rotation_;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.rotation_.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.preIntegration_->delta_v -
                                      frame_i->second.rotation_.transpose() * dt * Matrix3d::Identity() * g0;

            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, stateVectorNum - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(stateVectorNum - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        
        VectorXd dg = x.segment<2>(stateVectorNum - 3);
        
        g0 = (g0 + lxly * dg).normalized() * G.norm();
    }
    
    g = g0;
}

bool LinearAlignment(std::map<double, ImageFrame> &allImageFrame, Vector3d &g, VectorXd &x)
{
    int frameNum = allImageFrame.size();
    
    // (vx,vy,vz) for each frame, g_c0(gx,gy,gz), s 
    int stateVectorNum = frameNum * 3 + 3 + 1;

    MatrixXd A{stateVectorNum, stateVectorNum};
    A.setZero();
    VectorXd b{stateVectorNum};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = allImageFrame.begin(); next(frame_i) != allImageFrame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);
        
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();
        
        double dt = frame_j->second.preIntegration_->sum_dt;
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.rotation_.transpose() * dt * dt / 2 * Matrix3d::Identity();
        
        // set arbitrary initial scale s = 0.01
        tmp_A.block<3, 1>(0, 9) = frame_i->second.rotation_.transpose() * 
                                 (frame_j->second.translation_ - frame_i->second.translation_) / 100.0;     
        
        tmp_b.block<3, 1>(0, 0) = frame_j->second.preIntegration_->delta_p + frame_i->second.rotation_.transpose() *
                                  frame_j->second.rotation_ * TIC[0] - TIC[0];

        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.rotation_.transpose() * frame_j->second.rotation_;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.rotation_.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.preIntegration_->delta_v;

        // set information matrix as Identity
        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, stateVectorNum - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(stateVectorNum - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    

    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    
    // divide initial scale we set at first
    double s = x(stateVectorNum - 1) / 100.0;
    cout << "estimated scale: " << s << endl;
    
    g = x.segment<3>(stateVectorNum - 4);
    cout <<"result g: " << g.norm() << " " << g.transpose() << endl;
    
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
        return false;

    RefineGravity(allImageFrame, g, x);
    
    s = (x.tail<1>())(0) / 100.0;
    
    (x.tail<1>())(0) = s;
    
    cout << "refine g: " << g.norm() << " " << g.transpose() << endl;
    
    if(s < 0.0 )
        return false;   
    else
        return true;
}
    
bool VisualIMUAlignment(std::map<double, ImageFrame> &allImageFrame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(allImageFrame, Bgs);

    if(LinearAlignment(allImageFrame, g, x))
        return true;
    else 
        return false;
}


    
}
