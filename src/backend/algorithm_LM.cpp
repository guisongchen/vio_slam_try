#include "backend/algorithm_LM.h"
#include "backend/problem.h"

#include <math.h>
#include <iostream>
#include <iomanip>

namespace myslam
{
    
namespace backend
{
    
AlgorithmLM::AlgorithmLM(Problem* problem)
    : problem_(problem), currentLambda_(-1.0), tau_(1e-5),
      goodStepLowerScale_(1.0 / 3.0), goodStepUpperScale_(2.0 / 3.0),
      ni_(2.0), maxTrialsAfterFailure_(10), userLambda_(-1.0),
      currentChi2_(0.0), gainThreshold_(1e-6) {}


bool AlgorithmLM::Optimize(const int iterations)
{
    currentChi2_ = problem_->InitRobustChi2();
    ComputeInitLambda();
    
    bool report = problem_->ReportFlag();
    if (report)
    {
        std::cout << "----LM algorithm info----" << "\n"
                  << "Initial Lambda: " << currentLambda_ 
                  << "; Max Trials After Failure: " << maxTrialsAfterFailure_
                  << "; Gain Threshold: " << gainThreshold_ << "\n"
                  << "----optimizer info----" << std::endl;
    }
    
    bool stopFlag = false;
    int iter = 0;
    double lastChi2 = 0.0;
    
    while (!stopFlag && (iter < iterations)) 
    {
        bool acceptFlag = false;
        int trialCnt = 0;
        
        while (!acceptFlag && trialCnt < maxTrialsAfterFailure_)
        {
            problem_->SolveLinearSystem();
            
            problem_->UpdateStates();
            
            acceptFlag = CheckStepQuality();
            
            if (acceptFlag)
            {
                problem_->MakeHessian();
                trialCnt = 0;
            }
            else
            {
                trialCnt ++;
                problem_->RollbackStates();
            }
        }
        
        iter++;
        
        double gainChi2 = (lastChi2 - currentChi2_) / currentChi2_;
        
        if(std::fabs(gainChi2) < gainThreshold_ || currentChi2_ < 1e-6)
            stopFlag = true;
        
        lastChi2 = currentChi2_;
        
        if (report)
        {
            std::cout << "iter " << std::setw(3) << iter << ": "
                      << std::fixed << std::scientific << std::setprecision(6)
                      << "chi2 = " << currentChi2_ << "  "
                      << "lambda = " << currentLambda_ << std::endl;
        }
    }
    
    return stopFlag;
    
}


void AlgorithmLM::ComputeInitLambda()
{
    if (userLambda_ > 0)
        currentLambda_ = userLambda_;
    
    currentLambda_ = tau_ * problem_->MaxHessianDiagonal();
}

bool AlgorithmLM::CheckStepQuality()
{
    double predictGain = 0;
    
    Eigen::VectorXd delta_x = problem_->DeltaX();
    Eigen::VectorXd b = problem_->ErrorVector();
    
    predictGain = 0.5 * delta_x.transpose() * (currentLambda_ * delta_x + b);
    double updateChi2 = problem_->UpdateRobustChi2();
    double stepQuality = (currentChi2_ - updateChi2) / (predictGain + 1e-5);
    
    if (stepQuality > 0 && std::isfinite(updateChi2))
    {
        double alpha = 1.0 - std::pow((2 * stepQuality - 1), 3);
        alpha = std::min(alpha, goodStepUpperScale_);
        double lambdaScale = (std::max)(goodStepLowerScale_, alpha);
        
        currentLambda_ *= lambdaScale;
        ni_ = 2;
        currentChi2_ = updateChi2;
        
        return true;
    }
    else
    {
        currentLambda_ *= ni_;
        ni_ *= 2;
        
        return false;
    }
}

    
} // namespace backend

} // namespace myslam
