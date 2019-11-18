#include "system.h"
#include <thread>
#include <fstream>
#include <unistd.h>

const int delayTimes = 2;
std::string dataPath = "/home/dataset/EuRoC/MH_05_difficult/mav0/";
std::string configPath = "../config/";
std::shared_ptr<myslam::System> systemPtr;

void pubImuData()
{
    std::string imuDataFile = configPath + "imu_data.txt";
	cout << "PubImuData start imuDataFile: " << imuDataFile << endl;
    
	std::ifstream fImu;
	fImu.open(imuDataFile.c_str());
	if (!fImu.is_open())
	{
		std::cerr << "Failed to open imu file! " << imuDataFile << endl;
		return;
	}

	std::string imuLine;
	double stampNSec = 0.0;
	Vector3d acc;
	Vector3d gyr;
	while (std::getline(fImu, imuLine) && !imuLine.empty() && !systemPtr->ShutDown())
	{
		std::istringstream imuData(imuLine);
		imuData >> stampNSec >> gyr.x() >> gyr.y() >> gyr.z() >> acc.x() >> acc.y() >> acc.z();

		systemPtr->pubImuData(stampNSec/1e9, gyr, acc); // ns -> s
        
        // control publication frequency
		usleep(5000 * delayTimes); // 0.01s
	}
	
	fImu.close();
    cout << "IMU publisher shutdown" << endl;
}

void pubImgData()
{
    std::string imgDataFile = configPath + "cam_data.txt";
	cout << "PubImageData start imageDataFile: " << imgDataFile << endl;
    
	std::ifstream fImg;
	fImg.open( imgDataFile.c_str());
	if (!fImg.is_open())
	{
		std::cerr << "Failed to open image file! " << imgDataFile << endl;
		return;
	}

	std::string imgLine;
	double stampNSec = 0.0;
    std::string imgFileName;
	while (std::getline( fImg, imgLine ) && !imgLine.empty() && !systemPtr->ShutDown()) // read imu data
	{
		std::istringstream imgData ( imgLine );
		imgData >> stampNSec >> imgFileName;
        
        std::string imgPath = dataPath + "cam0/data/" + imgFileName;
        Mat img = cv::imread(imgPath.c_str(), 0);
        
        if (img.empty())
        {
            std::cerr << "image is empty! path: " << imgPath << endl;
			return;
        }
        
		systemPtr->pubImgData(stampNSec/1e9, img); // ns -> s
        
        // control publication frequency
		usleep(50000 * delayTimes); // 0.1s
	}
	
	fImg.close();
    
    cout << "Image publisher shutdown" << endl;
}

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		std::cerr << "./vio_run PATH_TO_FOLDER/MH-XX/mav0/ PATH_TO_CONFIG/config/" << endl; 
		return -1;
	}
	
	dataPath = argv[1];
	configPath = argv[2];

    systemPtr.reset(new myslam::System(configPath));
	
	std::thread thd_pubImuData(pubImuData);
	std::thread thd_pubImageData(pubImgData);
	std::thread thd_Draw(&myslam::System::Draw, systemPtr);
    systemPtr->processBackEnd();

	thd_pubImuData.join();
	thd_pubImageData.join();
	thd_Draw.join();

	cout << "main function finished, system shutdown." << endl;
	return 0;
}
