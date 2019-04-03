#include "loadVolLammpsDumpText.h"
#include <iostream>
#include <cassert>

struct i3_t
  {
    int x;
    int y;
    int z;
  };

struct f3_t
  {
    float x;
    float y;
    float z;
  };

int main(int argc, char** argv)
{
  if(argc != 3)
    {
      std::cout << "Usage: [exec] [dumpFile] [logFile]" << std::endl;
      return 0;
    }

  
  std::vector<float> vol;
  
  
  
  i3_t dim;

  std::string dumpFName(argv[1]);
  std::string logFName("");

  const int maxElemsPerDim = 64; 
  {
    f3_t bboxSubMin;
    f3_t bboxSubMax;
    
    int pId = 0;
    i3_t pGrid;
    pGrid.x = pGrid.y = pGrid.z = 1;
    bboxMinMaxFromPGrid(bboxSubMin, bboxSubMax, pId, pGrid);
    
    bool success = loadVolLammpsDumpText(vol, dim, dumpFName, maxElemsPerDim, bboxSubMin, bboxSubMax);
    assert(success);
  }

  {
    i3_t pGrid;
    bool success = pGridFromLammpsLog(pGrid, std::string(argv[2]));
    assert(success);
  }

  float total = 0.f;
  float maxEntry = 0.f;
  
  for(size_t i = 0; i<vol.size() ;i++)
    {
      maxEntry = std::max(maxEntry, vol[i]);
      total += vol[i];
    }

  std::cout << "maxEntry: " << maxEntry << " , total: " << total << std::endl;

  std::vector<unsigned char> volUChar(vol.size());

  for(size_t i = 0; i<vol.size() ;i++)
    volUChar[i] = (vol[i]/maxEntry)*255;

  {
    std::ofstream volFile;
    volFile.open ("volLammps.raw", std::ios::out | std::ios::binary);
    assert(volFile.is_open());

    volFile.write((char*) &volUChar[0], volUChar.size()*sizeof(unsigned char));  
    volFile.close();
  }
  {
    std::ofstream configFile;
    configFile.open ("volLammps.config", std::ios::out);
    assert(configFile.is_open());
    
    configFile << "VOLUME_FILE" << " " << "volData/volLammps.raw" << std::endl;
    configFile << "VOLUME_DIM " << dim.x << " " << dim.y << " " << dim.z << std::endl;
    configFile << "VOLUME_DATA_TYPE UCHAR" << std::endl;
    
    configFile.close();
  }

  return 0;
}
