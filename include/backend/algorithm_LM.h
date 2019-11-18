#ifndef MYSLAM_BACKEND_ALGORITHM_LM_H
#define MYSLAM_BACKEND_ALGORITHM_LM_H

namespace myslam
{
    
namespace backend
{
    
class Problem;
    
class AlgorithmLM
{
    
public:
    AlgorithmLM(Problem* problem);
    ~AlgorithmLM(){}
    
    bool Optimize(const int iterations);
    
    void ComputeInitLambda();
    
    double CurrentLambda() const
    {
        return currentLambda_;
    }
    
    double CurrentChi2() const
    {
        return currentChi2_;
    }
    
    void SetUserLambda(const double userLambda)
    {
        userLambda_ = userLambda;
    }
    
    double UserLambda() const
    {
        return userLambda_;
    }
    
    void SetGainThreshold(const double gainThreshold)
    {
        gainThreshold_ = gainThreshold;
    }
    
    double GainThreshold() const
    {
        return gainThreshold_;
    }
  
protected:
    
    Problem*        problem_;
    
    double          currentLambda_;
    double          tau_;
    double          goodStepLowerScale_;
    double          goodStepUpperScale_;
    double          ni_;
    int             maxTrialsAfterFailure_;

    double          userLambda_;
    double          currentChi2_;
    double          gainThreshold_;
    
    bool CheckStepQuality();
    
};

    
} // namespace backend

} // namespace myslam

#endif
