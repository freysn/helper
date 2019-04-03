#include "vec.h"
#include "transform.h"
#include "cimg_helper.h"
#include "VolDataHandlerCUDA.h"
#include "splitStr.h"
#include "volData.h"
#include "../helper/helper_SimpleTransFunc.h"
#include "points2vol.h"
#include <iomanip>


template<typename T, typename I3>
std::vector<T> run(std::vector<T>& in, I3 dim)
{
  std::vector<unsigned char> out;  

  //out = transform(in, dim, transformFunc1);
  out = transform(in, dim, transformFunc1_inv);

  return out;
}

template<typename T, typename I3>
void write(std::string fname, const T& out, I3 dim, int nChannels, bool isConfig)
{
  if(isConfig)
    {      
      if(dim.z==1)
        cimgWrite(fname+".png", out, dim, 1);

      writeFile(out, fname);
    }
  else
    {
      cimgWrite(fname+".png", out, dim, nChannels);
    }
}


int main(int argc, char** argv)
{
  if(argc != 3 && argc != 4)
    {
      std::cout << "wrong number of arguments\n";
      return 0;
    }  

  std::vector<unsigned char> in, out;

  int3 dim;
  int nChannels=1;

  const std::string strId = split(argv[1], '.').back();
  const bool isConfig =  (strId == "config");

  std::cout << "isConfig: " << isConfig << std::endl;
  if(isConfig)
    {
      VolDataHandlerCUDA<unsigned char> vdh(argv[1]);
      dim = vdh.getVolDim();
      assert(vdh.getNTimeSteps()==1);
      in = vdh[0];

      if(argc==4)
        {
          helper::SimpleTransFunc tf(argv[3]);
          const std::vector<unsigned char> b = tf.get();
          assert(b.size()==256);

          for(auto& e : in)
            e = b[e];

          writeFile(in, argv[2]);
          
          return 0;
        }

      if(false)
        {
          int3 dimNew = dim;
          dimNew.x*=2;
      
          std::vector<unsigned char> tmp(dimNew.x*dimNew.y*dimNew.z,0);

          int3 v;  
          for(v.z=0; v.z<dim.z; v.z++)
            for(v.y=0; v.y<dim.y; v.y++)
              for(v.x=0; v.x<dim.x; v.x++)
                {
                  tmp[iii2i(v, dimNew)] = in[iii2i(v, dim)];
                }
          dim = dimNew;
          in=tmp;
        }
    }
  else
    cimgRead(in, dim, nChannels, argv[1], true);

  //for(auto &e : in)
  //e /= 4;
  
  out = run(in, dim);
  write(argv[2], out, dim, nChannels, isConfig);

  size_t nT = 10;

  for(size_t t=0; t<=nT; t++)
    {
      const double w1 = ((double)t)/nT;
      const double w0 = 1.-w1;
      
      std::vector<unsigned char> outLin;

      {
        std::vector<std::pair<double3, double>> weightedPoints;

        auto createWeightedPoints = [&weightedPoints,&dim](const std::vector<unsigned char>& in, int3 dim, decltype(transformFunc1_inv) transFunc, double w0, double w1)
          {
            for(size_t i=0; i<in.size(); i++)
              {
                if(in[i]>0)
                  {
                    const double3 p = make<double>(i2iii(i, dim));
                    const double3 p2 = denormDim<double3>(transFunc(normDim(p, dim)), dim);

                    //std::cout << "p: " << p << std::endl;
                    //std::cout << "p2: " << p2 << std::endl;
                    
                    //assert(p.x >= 0. && p.y >= 0. && p.z >= 0.);
                    //assert(p2.x >= 0. && p2.y >= 0. && p2.z >= 0.);

                    
                    weightedPoints.emplace_back(w0*p+w1*p2,
                                                w0*in[i]);
                  }
              }
          };

        if(true)
          {
            createWeightedPoints(in, dim, transformFunc1_inv, w0, w1);
            createWeightedPoints(out, dim, transformFunc1, w1, w0);
          }
        else
          createWeightedPoints(in, dim, transformFunc1_inv, w0, 0.);

        std::vector<double> outLinf;
        splatWeightedPointsTri(outLinf, weightedPoints , dim);

        outLin.resize(outLinf.size());
        assert(outLin.size() == dim.x*dim.y*dim.z);
        for(size_t i=0; i<outLinf.size(); i++)
          outLin[i] = std::max(0, std::min((int)(outLinf[i]+0.5),255));
      }       

      std::stringstream ss;
      ss << "engine_lintrans_";
      ss << std::setw(5) << std::setfill('0') << t;

      
      if(isConfig)
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

      write(ss.str(), outLin, dim, nChannels, isConfig);
    }
  
}
