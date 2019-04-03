#include <iostream>
#include "downsample.h"
#include <random>
//#include <helper_math.h>
#include <iomanip>
#include "volData.h"
#include <climits>

#if 0

  struct int3
  {
    int x;
    int y;
    int z;
  };

  struct float3
  {
    float x;
    float y;
    float z;
  };
#else
#include <helper_math.h>
#endif

#include "helper_idx.h"
#include "VolDataHandlerCUDA.h"

#include "helper/helper_writeFile.h"

int main(int argc, char** argv)
{
  if(argc != 4)
    {
      std::cout << "Usage: downsample [config] [reduction factor] [out base]" << std::endl;
      return 0;
    }

  typedef unsigned char VolumeData_t;
  
  VolDataHandlerCUDA<VolumeData_t> vdh(argv[1]);
  const int reduction = std::stoi(argv[2]);
  
  const int3 dim = vdh.getVolDim();

  const std::string outBase(argv[3]);

  for(size_t t=0; t<vdh.getNTimeSteps(); t++)
    {
      int3 outDim = dim;
      std::vector<VolumeData_t> v;
      vdh.getH_c(t, 0, v, INT_MAX);
      std::vector<VolumeData_t> out;
      if(reduction > 1)
	{
#if 0
	  reduceLinear<double>(out, outDim, v, dim, reduction);
#else
	  outDim = dim;
	  outDim.x /= reduction;
	  outDim.y /= reduction;
	  outDim.z /= reduction;
	  int3 p;
	  out.resize(outDim.x*outDim.y*outDim.z);
	  for(p.z=0; p.z<outDim.z; p.z++)
	    for(p.y=0; p.y<outDim.y; p.y++)
	      for(p.x=0; p.x<outDim.x; p.x++)
		{
		  int3 d;

		  double m = 0.;
		  //VolumeData_t m = std::numeric_limits<VolumeData_t>::max();
		  for(d.z=0; d.z<reduction; d.z++)
		    for(d.y=0; d.y<reduction; d.y++)
		      for(d.x=0; d.x<reduction; d.x++)
			{
			  const auto e = v[iii2i_clamp(p*reduction+d, dim)];
			  //m = std::min(m, e);
			  m += e;
			}
		  out[iii2i(p, outDim)] = m/(reduction*reduction*reduction);
		}		
#endif
	  
	}
      else
        out = v;

      std::stringstream ss;
      ss << outBase << std::setw(5) << std::setfill('0') << t;
      
      helper::writeFile(out, ss.str());
      std::cout << "write " << ss.str()
                << " | dim " << outDim.x << " " << outDim.y << " " << outDim.z << std::endl;
    }
  
  return 0;
}
