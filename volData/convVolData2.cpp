#include "volData.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <fstream>

//typedef float T_in;
typedef unsigned short T_in;
typedef unsigned char T_out;
//typedef unsigned short T_out;

//const double valueMaxOut = 65535.f;
const double valueMaxOut = 255.f;

int main(int argc, char** argv)
{
  const int reductionPerDim = 1;
  //m_uint3 fromDim(720, 320, 320);
  //m_uint3 fromDim(1024, 1024, 1080);
  //m_uint3 fromDim(128, 128, 256);
  //m_uint3 fromDim(680, 680, 680);
  m_uint3 fromDim(442, 442, 1500);
  m_uint3 toDim = fromDim;
  toDim.x /= reductionPerDim;
  toDim.y /= reductionPerDim;
  toDim.z /= reductionPerDim;

  
  
  //char* filename = "/data/volumes/volume/utct/oldfield_mouse.raw";
  std::string filename(argv[1]);

  const size_t nElemsIn = fromDim.x*fromDim.y*fromDim.z;
  const size_t nElemsOut = toDim.x*toDim.y*toDim.z;
  
  T_in* fromData = new T_in[nElemsIn];
  T_in* toDataBuf = new T_in[nElemsOut];
  
  T_out* toData = new T_out[nElemsOut];

  double minVal = DBL_MAX;
  double maxVal = -DBL_MAX;

  const unsigned int numFixedLen = 3;

  const int filterKernelDim = 5;
  //const int filterKernelDim = 3;
  
  for(int p=0; p<=1; p++)
    {
      //for(int i=0; i<=41; i++)
      for(int i=0; i<25; i++)
        { 
#if 0
          std::stringstream ss;
          ss << setw(numFixedLen) << setfill('0') << i;
      
          std::string baseName = filename+ss.str(); 
          baseName += ".raw";
#else          
          std::string baseName = filename; 
#endif
          std::cout << baseName << std::endl;

          std::ifstream fin;
          fin.open((char*)baseName.c_str(), std::ios::in | std::ios::binary);
          assert(fin.is_open());
          fin.read((char*)fromData, sizeof(T_in)*nElemsIn);
      
      
          //unsigned short* fromData = 
          //loadRawFile((char*)(baseName+".raw").c_str(), fromDim.x*fromDim.y*fromDim.z, elemType);
                
          if(p==0)
            {
              for(size_t j=0; j<nElemsIn; j++)
                {
                  double v = (double)(fromData[j]);
                  minVal = std::min(minVal, v);
                  maxVal = std::max(maxVal, v);
                }
            }
          else if(p==1)
            {
              
              for(int z=0; z<fromDim.z; z+=reductionPerDim)
                for(int y=0; y<fromDim.y; y+=reductionPerDim)
                  for(int x=0; x<fromDim.x; x+=reductionPerDim)
                {
                  double normValue = 0.f;
                  
                  for(int rz=0; rz<reductionPerDim; rz++)
                    for(int ry=0; ry<reductionPerDim; ry++)
                      for(int rx=0; rx<reductionPerDim; rx++)
                        {
                          const size_t idx = x+rx + fromDim.x*(y+ry+fromDim.y*(z+rz));
                          normValue += (fromData[idx]-minVal)/(maxVal-minVal);
                        }
                  
                  normValue /= (reductionPerDim*reductionPerDim*reductionPerDim);
                  
                  const size_t idx = 
                    x/reductionPerDim + 
                    toDim.x*(y/reductionPerDim + toDim.y*(z/reductionPerDim));
                  //toDataBuf[idx] = powf(normValue, 4)*255.f;
                  toDataBuf[idx] = normValue*valueMaxOut;
                  toData[idx] = toDataBuf[idx];
                }

              
#if 0
              for(int z=filterKernelDim/2; z<toDim.z-filterKernelDim/2; z++)
                for(int y=filterKernelDim/2; y<toDim.y-filterKernelDim/2; y++)
                  for(int x=filterKernelDim/2; x<toDim.x-filterKernelDim/2; x++)
                    {
                      double totalWeight = 0.f;
                      double totalValue = 0.f;
                      for(int rz = -filterKernelDim/2; rz<filterKernelDim/2; rz++)
                        for(int ry = -filterKernelDim/2; ry<filterKernelDim/2; ry++)
                          for(int rx = -filterKernelDim/2; rx<filterKernelDim/2; rx++)
                            {
                              double weight = 4.f;
                              if(rx!=0 || ry!=0 || rz!=0)
                                weight = 1./(double)(abs(rx)+abs(ry)+abs(rz));
                              totalWeight += weight;
                              totalValue += weight*toDataBuf[x+rx+toDim.x*(y+ry+toDim.y*(z+rz))];                              
                            }
                      assert(totalWeight > 0.f);
                      toData[x+toDim.x*(y+toDim.y*(z))] = totalValue/totalWeight;
                    }
#endif              
              
              std::stringstream ss;
              ss << setw(numFixedLen) << setfill('0') << i;
              
              writeFile(toData, toDim.x*toDim.y*toDim.z, 
                        (char*)(filename+"_UCHAR_"+ss.str()+".raw").c_str());
            }
      
        }
      std::cout << "minVal: " << minVal << std::endl;
      std::cout << "maxVal: " << maxVal << std::endl;
    }
  
  return 0;
}

