#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if(argc != 3)
	{
		std::cerr << "usage: ./euroc2tum PATH_TO_CSV_FILE SAVE_AS_NAME" << std::endl; 
		return -1;
	}
	
	std::string csvPath = argv[1];
	std::string saveName = argv[2];
    
    std::ifstream csvFile;
	csvFile.open(csvPath.c_str());
    if (!csvFile.is_open())
	{
		std::cerr << "Failed to open csv file! check path: " << csvPath << std::endl;
		return -1;
	}
	
	std::ofstream txtFile;
    txtFile.open("../config/" + saveName, std::fstream::out);
    if (!txtFile.is_open())
        std::cerr << "txt file is not open" << std::endl;
    
    std::string csvLine;
    while (std::getline(csvFile, csvLine) && !csvLine.empty())
	{
        if (csvLine.at(0) == '#')
            continue;
        
        for (auto &i : csvLine)
        {
            if (i == ',')
                i = ' ';
        }
        
        txtFile << csvLine << std::endl;
	}
	
	csvFile.close();
    txtFile.close();
    
}
