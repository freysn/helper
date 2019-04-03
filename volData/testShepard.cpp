#include "vec.h"
#include "shepard.h"
#include "VolDataHandlerCUDA.h"
#include "volData.h"
#include "transform.h"
#include "shepardLocallyAffine.h"


int main(int argc, char** argv)
{

  //const std::string configFName = "configs/engine.config";
  const std::string configFName = "configs/intervol/transvsmorph/engine_mapped.config";
  
  const std::string cornersFName = "../..//Dropbox/dev/intervol_doc/reviews/peaks";
  //const std::string configFName = "configs/engine_mapped_thin.config";

  //const std::string cornersFName = "../../Dropbox/dev/intervol_doc/reviews/peaks_slice";
  //const std::string configFName = "configs/engine_slice.config";

  //const std::string configFName = "engine_lintrans_00000.config";
  //const std::string configFName = "configs/engine_slice.config";
  //const std::string configFName = "configs/cuboid_slice.config";

  auto corners0 = readCorners<double3>(cornersFName);
  
    
  if(corners0.empty())
    {
      std::cout << "could not find file " << cornersFName << std::endl;
      return 0;
    }

  VolDataHandlerCUDA<unsigned char> vdh(configFName);
  const auto dim = vdh.getVolDim();
  assert(vdh.getNTimeSteps()==1);
  const auto in = vdh[0];

  /*
  corners.push_back(make_int3(0,0,0));
  corners.push_back(make_int3(dim.x-1,0,0));
  corners.push_back(make_int3(dim.x,dim.y-1,0));
  corners.push_back(make_int3(0,dim.y-1,0));
  */

  auto corners1 = corners0;
  for(auto &e : corners1)
    e = denormDim<double3>(transformFunc1_inv(normDim(e,dim)), dim);
    
  size_t nT = 10;

  for(size_t t=1; t<=nT; t++)
    {
      std::cout << "processing " << t << " of " << nT << std::endl;
      auto corners1t = corners1;
      const double w1 = ((double)t)/nT;
      const double w0 = 1.-w1;
      
      for(size_t i=0; i<corners1.size(); i++)
        {
          corners1t[i] =
            w0*corners0[i]+w1*corners1[i];
        }
      
      const auto out =
        //shepard(in, dim, corners0, corners1)
        shepardLocallyAffine(in, dim, corners0, corners1t)
        ;

      std::stringstream ss;
      ss << "engine_shepard_";
      ss << std::setw(5) << std::setfill('0') << t;
      
      writeFile(out, ss.str());

      {
        UserData ud;
        ud._volumeFiles.resize(1, "../intervol/"+ss.str());
        ud._volDim.x = dim.x;
        ud._volDim.y = dim.y;
        ud._volDim.z = dim.z;

        /*
        ud._voxelSize.x = voxelSize.x;
        ud._voxelSize.y = voxelSize.y;
        ud._voxelSize.z = voxelSize.z;
        */
        
        const bool success = ud.writeConfig(ss.str()+".config");
        assert(success);
      }

      {
        std::ofstream fout(ss.str()+"_peaks");
        assert(fout.is_open());
        for(auto c : corners1t)
          fout << c.x << " " << c.y << " " << c.z << std::endl;
      }

      if(dim.z==1)
        {
          if(t==1)
            cimgWrite("shepard_in.png", in, dim, 1);
          cimgWrite(ss.str()+".png", out, dim, 1);
        }
    }
  
  return 0;
}
