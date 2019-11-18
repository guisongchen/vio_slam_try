#include "parameter.h"
#include <opencv2/core/eigen.hpp>


double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;

std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;

// string IMU_TOPIC;
double ROW, COL;
double TD, TR;



// int FOCAL_LENGTH;
std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
// int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
bool STEREO_TRACK;
int EQUALIZE;
int FISHEYE;
bool PUB_THIS_FRAME;



void readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "readParameters ERROR: Wrong path to settings!" << endl;
        return;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.txt";

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        // ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0){
            cout << "fix extrinsic param " << endl;
        }
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
    }
    else
    {
        TR = 0;
    }

    //string VINS_FOLDER_PATH = readParam<string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    // if (FISHEYE == 1)
    //     FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    // WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    PUB_THIS_FRAME = false;

    if (FREQ == 0){
        FREQ = 10;
    }
    fsSettings.release();

    cout << "readParameters:  "
        <<  "\n  INIT_DEPTH: " << INIT_DEPTH
        <<  "\n  MIN_PARALLAX: " << MIN_PARALLAX
        <<  "\n  ACC_N: " <<ACC_N
        <<  "\n  ACC_W: " <<ACC_W
        <<  "\n  GYR_N: " <<GYR_N
        <<  "\n  GYR_W: " <<GYR_W
        <<  "\n  RIC:   " << RIC[0]
        <<  "\n  TIC:   " <<TIC[0].transpose()
        <<  "\n  G:     " <<G.transpose()
        <<  "\n  BIAS_ACC_THRESHOLD:"<<BIAS_ACC_THRESHOLD
        <<  "\n  BIAS_GYR_THRESHOLD:"<<BIAS_GYR_THRESHOLD
        <<  "\n  SOLVER_TIME:"<<SOLVER_TIME
        <<  "\n  NUM_ITERATIONS:"<<NUM_ITERATIONS
        <<  "\n  ESTIMATE_EXTRINSIC:"<<ESTIMATE_EXTRINSIC
        <<  "\n  ESTIMATE_TD:"<<ESTIMATE_TD
        <<  "\n  ROLLING_SHUTTER:"<<ROLLING_SHUTTER
        <<  "\n  ROW:"<<ROW
        <<  "\n  COL:"<<COL
        <<  "\n  TD:"<<TD
        <<  "\n  TR:"<<TR
        <<  "\n  FOCAL_LENGTH:"<<FOCAL_LENGTH
        <<  "\n  IMAGE_TOPIC:"<<IMAGE_TOPIC
        <<  "\n  IMU_TOPIC:"<<IMU_TOPIC
        <<  "\n  FISHEYE_MASK:"<<FISHEYE_MASK
        <<  "\n  CAM_NAMES[0]:"<<CAM_NAMES[0]
        <<  "\n  MAX_CNT:"<<MAX_CNT
        <<  "\n  MIN_DIST:"<<MIN_DIST
        <<  "\n  FREQ:"<<FREQ
        <<  "\n  F_THRESHOLD:"<<F_THRESHOLD
        <<  "\n  SHOW_TRACK:"<<SHOW_TRACK
        <<  "\n  STEREO_TRACK:"<<STEREO_TRACK
        <<  "\n  EQUALIZE:"<<EQUALIZE
        <<  "\n  FISHEYE:"<<FISHEYE
        <<  "\n  PUB_THIS_FRAME:"<<PUB_THIS_FRAME
    << endl;
}
