#include "initial/initial_sfm.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <memory>
#include <unordered_map>

#include "backend/problem.h"
#include "backend/algorithm_LM.h"
#include "backend/vertex_types.h"
#include "backend/edge_types.h"

namespace myslam
{

bool GlobalSFM::construct(int frameNum, Quaterniond* quatsRefWorld, Vector3d* transRefWorld, int refFrameId,
                          const Matrix3d Rrc, const Vector3d trc,
                          std::vector<SFMFeature> &sfmFeatures, std::map<int, Vector3d> &sfmTrackedPoints)
{
    featureNum_ = sfmFeatures.size();
    
	// set reference frame as world frame
    quatsRefWorld[refFrameId].w() = 1;
    quatsRefWorld[refFrameId].x() = 0;
    quatsRefWorld[refFrameId].y() = 0;
    quatsRefWorld[refFrameId].z() = 0;
    transRefWorld[refFrameId].setZero();
    
    const int currFrameId = frameNum - 1;
    quatsRefWorld[currFrameId] = quatsRefWorld[refFrameId] * Quaterniond(Rrc);
    transRefWorld[currFrameId] = trc;

	//rotation and translation of all frames in camera frame(coordinate)
	Matrix3d rotsCam[frameNum];
	Vector3d transCam[frameNum];
	Quaterniond quatsCam[frameNum];
	Eigen::Matrix<double, 3, 4> Pose[frameNum];
    
    quatsCam[refFrameId] = quatsRefWorld[refFrameId].inverse();
    rotsCam[refFrameId] = quatsCam[refFrameId].toRotationMatrix();
    transCam[refFrameId] = -1 * (rotsCam[refFrameId] * transRefWorld[refFrameId]);
	Pose[refFrameId].block<3, 3>(0, 0) = rotsCam[refFrameId];
	Pose[refFrameId].block<3, 1>(0, 3) = transCam[refFrameId];
    
    quatsCam[currFrameId] = quatsRefWorld[currFrameId].inverse();
    rotsCam[currFrameId] = quatsCam[currFrameId].toRotationMatrix();
    transCam[currFrameId] = -1 * (rotsCam[currFrameId] * transRefWorld[currFrameId]);
    Pose[currFrameId].block<3, 3>(0, 0) = rotsCam[currFrameId];
	Pose[currFrameId].block<3, 1>(0, 3) = transCam[currFrameId];

	//1: trangulate between refFrameId ----- currFrameId-1
	//2: solve pnp refFrameId + 1; trangulate refFrameId + 1 ------- currFrame; 
	for (int i = refFrameId; i < currFrameId; i++)
	{
		// solve pnp
		if (i > refFrameId)
		{
			// use (i-1)th frame pose as initial pose of i-th frame
            Matrix3d R_iw = rotsCam[i - 1];
			Vector3d t_iw = transCam[i - 1];
			
            // solve for optimized pose
            if(!solveFrameByPnP(R_iw, t_iw, i, sfmFeatures))
				return false;
            
            // update pose
            rotsCam[i] = R_iw;
            transCam[i] = t_iw;
            quatsCam[i] = rotsCam[i];
			Pose[i].block<3, 3>(0, 0) = rotsCam[i];
			Pose[i].block<3, 1>(0, 3) = transCam[i];
		}

		// triangulate point based on the solve pnp result
		// triangulate between i and currFrame, i from refFrame to currFrame-1.
		// we got initial 3d position from refFrame and currFrame, then used to solve pnp
		triangulateTwoFrames(i, Pose[i], currFrameId, Pose[currFrameId], sfmFeatures);
	}
	
	//3: if some frame between refId to currId are NOT triangulated(solvepnp failed), do it now
	//   but this time, use reference frame as reference, not current frame
	for (int i = refFrameId + 1; i < currFrameId; i++)
		triangulateTwoFrames(refFrameId, Pose[refFrameId], i, Pose[i], sfmFeatures);
	
    //4: till now, 0 to refId position are NOT triangulated, do it 
	for (int i = refFrameId - 1; i >= 0; i--)
	{
		// use (i+1)th frame pose as initial pose of i-th frame, since i-- here
		Matrix3d R_iw = rotsCam[i + 1];
		Vector3d t_iw = transCam[i + 1];
        
		if(!solveFrameByPnP(R_iw, t_iw, i, sfmFeatures))
			return false;
        
        rotsCam[i] = R_iw;
        transCam[i] = t_iw;
        quatsCam[i] = rotsCam[i];
		Pose[i].block<3, 3>(0, 0) = rotsCam[i];
		Pose[i].block<3, 1>(0, 3) = transCam[i];
        
		// triangulate between i and refFrame, i from refFrame-1 to 0. 
		triangulateTwoFrames(i, Pose[i], refFrameId, Pose[refFrameId], sfmFeatures);
	}
	
	//5: since all frames poses are solved, triangulate remain points observed by more than 2 frames
	for (int i = 0; i < featureNum_; i++)
	{
		if (sfmFeatures[i].triangulateState_) // has been triangulated
			continue;
        
		if (static_cast<int>(sfmFeatures[i].observation_.size()) >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfmFeatures[i].observation_[0].first;
			point0 = sfmFeatures[i].observation_[0].second;
			int frame_1 = sfmFeatures[i].observation_.back().first;
			point1 = sfmFeatures[i].observation_.back().second;
			Vector3d point_3d;
            
            // triangulate between first and last observed frame 
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            
            sfmFeatures[i].triangulateState_ = true;
            sfmFeatures[i].position_[0] = point_3d(0);
            sfmFeatures[i].position_[1] = point_3d(1);
            sfmFeatures[i].position_[2] = point_3d(2);
		}		
	}
    
    backend::Problem* problem(new backend::Problem());
    backend::AlgorithmLM* algorithmLM(new backend::AlgorithmLM(problem));
    problem->SetAlgorithm(algorithmLM);

    std::vector<std::shared_ptr<backend::VertexPose> > vertexCams;
    std::unordered_map<int, std::shared_ptr<backend::VertexPointXYZ> > vertexPoints;
    
	for (int i = 0; i < frameNum; i++)
	{
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        double pose[7];
        memcpy(pose, transCam[i].data(), 3 * sizeof(double));
        memcpy(pose + 3, quatsCam[i].coeffs().data(), 4 * sizeof(double));
        vertexCam->SetParameterData(pose);
        
        vertexCams.push_back(vertexCam);
        problem->AddVertex(vertexCam);
	}
	
	Eigen::Matrix2d identityInfo(Eigen::Matrix2d::Identity());
    
	for (int i = 0; i < featureNum_; i++)
	{
		// if not triangulated, continue
        if (!sfmFeatures[i].triangulateState_)
			continue;

        std::shared_ptr<backend::VertexPointXYZ> vertexPoint(new backend::VertexPointXYZ());
        vertexPoint->SetParameterData(sfmFeatures[i].position_);
        
        vertexPoints.insert(std::make_pair(i, vertexPoint));
        problem->AddVertex(vertexPoint);
        
        const int obsNum = static_cast<int>(sfmFeatures[i].observation_.size());
        for (int j = 0; j < obsNum; j++)
		{
			int frameId = sfmFeatures[i].observation_[j].first;
            Vector2d obs = sfmFeatures[i].observation_[j].second;
            
            std::shared_ptr<backend::EdgeReprojectXYZ> edge(new backend::EdgeReprojectXYZ(obs));
            edge->AddVertex(vertexCams[frameId]);
            edge->AddVertex(vertexPoint);
            edge->SetInformation(identityInfo);
            edge->SetCameraIntrisic(1.0, 1.0, 0.0, 0.0);
            
            problem->AddEdge(edge);
		}
	}

    problem->SetReport(true);
    bool result = problem->SolveProblem(100);
    
    if (result || problem->FinalCost() < 1e-04)
        cout << "global sfm successed! final cost: " << problem->FinalCost() << endl;
    else
    {
        cout << "global sfm failed!" << endl;
        return false;
    }
	
	for (int i = 0; i < frameNum; i++)
	{
        Eigen::VectorXd pose = vertexCams[i]->Parameters();
        Quaterniond quatCam(pose[6], pose[3], pose[4], pose[5]);
        Vector3d transCam(pose[0], pose[1], pose[2]);

        quatsRefWorld[i] = quatCam.inverse();
        transRefWorld[i] = -1 * (quatsRefWorld[i] * transCam);
	}
	
    // get feature 3d postion 
    for (int i = 0; i < featureNum_; i++)
	{
		if(!sfmFeatures[i].triangulateState_)
            continue;

        sfmTrackedPoints[sfmFeatures[i].id_] = vertexPoints[i]->Parameters();
	}
	
	delete algorithmLM;
    delete problem;
	
	return true;
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_iw, Vector3d &t_iw, int targetFrameId,
								std::vector<SFMFeature> &sfmFeatures)
{
	std::vector<cv::Point2f> p2ds;
	std::vector<cv::Point3f> p3ds;
	for (int i = 0; i < featureNum_; i++)
	{
		if (!sfmFeatures[i].triangulateState_)
			continue;
        
        const int obsNum = static_cast<int>(sfmFeatures[i].observation_.size());
		for (int k = 0; k < obsNum; k++)
		{
			// if current feature was tracked by targetFrame
            // 3d postions come from triangulate between refFrame and currFrame
            if (sfmFeatures[i].observation_[k].first == targetFrameId)
			{
				Vector2d &img_pts = sfmFeatures[i].observation_[k].second;
				cv::Point2f p2d(img_pts[0], img_pts[1]);
                p2ds.push_back(p2d);
                
				cv::Point3f p3d(sfmFeatures[i].position_[0], sfmFeatures[i].position_[1], sfmFeatures[i].position_[2]);
                p3ds.push_back(p3d);
                
				break;
			}
		}
	}
	
	const int matchedNum = static_cast<int>(p2ds.size());
	
	if (matchedNum < 15)
	{
		cout << "unstable features tracking, please slowly move you device!" << endl;
		if (matchedNum < 10)
			return false;
	}
	
	cv::Mat r, rvec, tiw_mat, Distort, Riw_mat;
	cv::eigen2cv(R_iw, Riw_mat);
	cv::Rodrigues(Riw_mat, rvec);
	cv::eigen2cv(t_iw, tiw_mat);
    
	// why use this intrinsic matrix?
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    
	bool pnpSuccFlag;
    
    // use r, t as initial value for useExtrinsicGuess = 1
    pnpSuccFlag = cv::solvePnP(p3ds, p2ds, K, Distort, rvec, tiw_mat, 1);
	
    if(!pnpSuccFlag)
		return false;
    
	cv::Rodrigues(rvec, r);
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd t_pnp;
	cv::cv2eigen(tiw_mat, t_pnp);
    R_iw = R_pnp;
    t_iw = t_pnp;
    
	return true;
}

void GlobalSFM::triangulateTwoFrames(int frameId0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frameId1, Eigen::Matrix<double, 3, 4> &Pose1,
									 std::vector<SFMFeature> &sfmFeatures)
{
	assert(frameId0 != frameId1);
    
	for (int i = 0; i < featureNum_; i++)
	{
		if (sfmFeatures[i].triangulateState_)
			continue;
        
		bool observed0 = false, observed1 = false;
		Vector2d point0;
		Vector2d point1;
        
        const int obsNum = static_cast<int>(sfmFeatures[i].observation_.size());
		for (int k = 0; k < obsNum; k++)
		{
			if (sfmFeatures[i].observation_[k].first == frameId0)
			{
                point0 = sfmFeatures[i].observation_[k].second;
                observed0 = true;
			}
			
			if (sfmFeatures[i].observation_[k].first == frameId1)
			{
                point1 = sfmFeatures[i].observation_[k].second;
                observed1 = true;
			}
		}
		
		if (observed0 && observed1)
		{
			Vector3d p3d;
			triangulatePoint(Pose0, Pose1, point0, point1, p3d);
            sfmFeatures[i].triangulateState_ = true;
            sfmFeatures[i].position_[0] = p3d(0);
            sfmFeatures[i].position_[1] = p3d(1);
            sfmFeatures[i].position_[2] = p3d(2);
		}							  
	}
}

// Trianularization same as ORB_SLAM
// x' = P'X  x = PX
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|

// DLT：[x]xPX = 0

// for point x
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|

// same for point x'
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|

// swith row squence and multiply -1 for some row, we get:
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1, Vector3d &p3d)
{
	Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    
	Eigen::Vector4d triangulated_point;
	triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	
    // homogeneous to non-homogeneous
    p3d(0) = triangulated_point(0) / triangulated_point(3);
    p3d(1) = triangulated_point(1) / triangulated_point(3);
    p3d(2) = triangulated_point(2) / triangulated_point(3);
}

}
