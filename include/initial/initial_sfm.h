#ifndef INITIAL_SFM_H
#define INITIAL_SFM_H

#include "common_include.h"

namespace myslam
{
    
struct SFMFeature
{
    bool    triangulateState_;
    int     id_;
    double  position_[3];
    double  depth_;
    
    // frameId, x,y of normalized plane
    std::vector<std::pair<int,Vector2d> > observation_;
};


class GlobalSFM
{
public:
	GlobalSFM() {}
	bool construct(int frameNum, Quaterniond* quatsRefWorld, Vector3d* transRefWorld, int refFrameId,
                   const Matrix3d Rrc, const Vector3d trc,
                   std::vector<SFMFeature> &sfmFeatures, std::map<int, Vector3d> &sfmTrackedPoints);

private:
	bool solveFrameByPnP(Matrix3d &R_iw, Vector3d &t_iw, int targetFrameId, std::vector<SFMFeature> &sfmFeatures);
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                          Vector2d &point0, Vector2d &point1, Vector3d &p3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  std::vector<SFMFeature> &sfmFeatures);

	int featureNum_;
};
    
}

#endif
