#include <iostream>
#include "downsample.h"
#include <random>
//#include <helper_math.h>
#include <iomanip>
#include "volData.h"
#include <climits>
#include "vec.h"


#include "helper_readFile.h"

int main(int argc, char** argv)
{
  if(argc != 3)
    {
      std::cout << "Usage: downsample [config] [out base]" << std::endl;
      return 0;
    }

  //typedef unsigned char VolumeData_t;
  typedef float3 VolumeData_t;

  UserData userData;
  userData.readConfig(argv[1]);
  
  
  const auto dim = userData._volDim;

  const std::string outBase(argv[2]);
  std::vector<VolumeData_t> v;
  
  for(size_t t=0; t<userData._nTimeSteps; t++)
    {
      
      size_t nElems = helper::readFile_FS3D(v, userData.getFileName(t),
					    userData);
      std::cout << nElems << " vs " << v.size() << " vs " << dim.x*dim.y*dim.z << std::endl;
      
      std::stringstream ss;
      ss << outBase << std::setw(5) << std::setfill('0') << t;
      
      writeFile(v, ss.str());
      std::cout << "write " << ss.str()
                << " | dim " << dim.x << " " << dim.y << " " << dim.z << std::endl;
    }
  
  return 0;
}
