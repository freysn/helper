#include <iostream>
#include "downsample.h"
#include <random>
//#include <helper_math.h>
#include <iomanip>
#include "volData.h"
#include <climits>
#include "vec.h"


#include "helper_readFile.h"
#include "read_traw3D_ss.h"

#include <algorithm>

int main(int argc, char** argv)
{
  if(argc != 2)
    {
      std::cout << "Usage: downsample [config]" << std::endl;
      return 0;
    }

  std::vector<short> volume;
  int3 volDim;
  float3 voxelSize;
  const std::string fname(argv[1]);
  
  const bool success = read_traw3D_ss(volume, volDim, voxelSize, fname);

  assert(success);
  std::cout << "volDim: " << volDim.x << " " << volDim.y << " " << volDim.z << std::endl;
  std::cout << "voxelSize: " << voxelSize.x << " " << voxelSize.y << " " << voxelSize.z << std::endl;
  

  auto bounds = std::minmax_element(volume.begin(), volume.end());

  std::cout << "min: " << (*bounds.first) << " max: " << (*bounds.second) << std::endl;

  std::vector<unsigned char> v(volume.size());
  std::vector<size_t> hist(256, 0);
  for(size_t i=0; i<v.size(); i++)
    {
      const int e = 255.*static_cast<double>(volume[i]-(*bounds.first))
        /((*bounds.second)-(*bounds.first));
      assert(e>=0 && e<256);
      hist[e]++;
      v[i] = e;        
    }  

  std::cout << "histogram:\n";
  for(size_t i=0;i<hist.size(); i++)
    std::cout << i << " " << 100.*static_cast<double>(hist[i])/v.size() << "\%" << std::endl;
    
  {
    const std::string fname = std::string(argv[1])+".raw";
  
    writeFile(v, fname);
    std::cout << "write " << fname
              << std::endl;
  }
  
  return 0;
}
