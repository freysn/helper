#ifndef __RESAMPLE__
#define __RESAMPLE__

#include <limits>
#include <vector>
#include <iostream>
#include <map>
#include <helper_math.h>
#include <cassert>
#include "downsample.h"


#if 1
template<typename V3, typename S, typename T, typename I3>
  void resample(std::vector<T>& outData, I3& dim, std::vector<std::pair<V3, S>>& pointsValues,  size_t maxResDim)
{
  
  

  V3 bboxMin = pointsValues[0].first;  
  V3 bboxMax = pointsValues[0].first;
  
  for(int64_t i=0; i<pointsValues.size(); i++)
    {
      bboxMin.x = std::min(pointsValues[i].first.x, bboxMin.x);
      bboxMin.y = std::min(pointsValues[i].first.y, bboxMin.y);
      bboxMin.z = std::min(pointsValues[i].first.z, bboxMin.z);

      bboxMax.x = std::max(pointsValues[i].first.x, bboxMax.x);
      bboxMax.y = std::max(pointsValues[i].first.y, bboxMax.y);
      bboxMax.z = std::max(pointsValues[i].first.z, bboxMax.z);

    }

  std::cout << "boxMin: " << bboxMin.x << " " << bboxMin.y << " " << bboxMin.z << std::endl;
  std::cout << "boxMax: " << bboxMax.x << " " << bboxMax.y << " " << bboxMax.z << std::endl;
  {
    V3 dv = bboxMax-bboxMin;
    double dvmax = std::max(dv.x, std::max(dv.y, dv.z));
    dim.x = (dv.x/dvmax)*maxResDim;
    dim.y = (dv.y/dvmax)*maxResDim;
    dim.z = (dv.z/dvmax)*maxResDim;
  }

  std::cout << "dim " << dim.x<< " " << dim.y << " " << dim.z << std::endl;
  
  outData.resize(dim.x*dim.y*dim.z, 0.14);
  
  std::vector<double> dists(outData.size(), std::numeric_limits<double>::max());
  
  //std::cout << "insert point values in grd\n";
  for(int64_t i=0; i<pointsValues.size(); i++)
    {
      V3 p = pointsValues[i].first;      
      V3 np = (p-bboxMin)/(bboxMax-bboxMin);      
      V3 nps;
      nps.x = np.x*dim.x;
      nps.y = np.y*dim.y;
      nps.z = np.z*dim.z;
      
      I3 pos;
      pos.x = std::max(0, std::min(static_cast<int>(nps.x), dim.x));
      pos.y = std::max(0, std::min(static_cast<int>(nps.y), dim.y));
      pos.z = std::max(0, std::min(static_cast<int>(nps.z), dim.z));

      for(int64_t z=pos.z; z<=std::min(pos.z+1, dim.z-1); z++)
        for(int64_t y=pos.y; y<=std::min(pos.y+1, dim.y-1); y++)
          for(int64_t x=pos.x; x<=std::min(pos.x+1, dim.x-1); x++)
            {
              V3 gp;
              gp.x = x;
              gp.y = y;
              gp.z = z;
              const V3 dv = gp-nps;
              const double d = dot(dv, dv);

              const int64_t e = x+dim.x*(y+dim.y*z);
              if(d < dists[e])
                {
                  dists[e] = d;
                  outData[e] = pointsValues[i].second;
                }
              
            }
      
    }
  
  std::multimap<double, int3> deltaVec;
  
  //std::cout << "construct delta vec\n";
  
  
  {
    int rad = 2;
    int3 dv;
    int64_t cnt=0;
    for(dv.z=-rad; dv.z<=rad; dv.z++)
      {
        //std::cout << "z: "<< dv.z << std::endl;
      for(dv.y=-rad; dv.y<=rad; dv.y++)
        for(dv.x=-rad; dv.x<=rad; dv.x++)
          {
            double d = (double)dv.x*dv.x+dv.y*dv.y+dv.z*dv.z;
            deltaVec.insert(std::make_pair(d,dv));
            cnt++;
          }
      }
  }
    
  //std::cout << "fill holes\n";
  for(int64_t z=0; z<dim.z; z++)
    {
      //std::cout << "z: "<< z << std::endl;
    for(int64_t y=0; y<dim.y; y++)
      for(int64_t x=0; x<dim.x; x++)
        {
          std::multimap<double, int3>::const_iterator dvIt =  deltaVec.begin();
          const int64_t eidx = x+dim.x*(y+dim.y*z);
          double d = dists[eidx];
          int64_t idx = eidx;
          while(d == std::numeric_limits<double>::max())
            {
              
              idx =
                std::max((int64_t)0, std::min(x+dvIt->second.x, (int64_t)dim.x-1))+
                dim.x*(std::max((int64_t)0, std::min(y+dvIt->second.y, (int64_t)dim.y-1))+
                       dim.y*(std::max((int64_t)0, std::min(z+dvIt->second.z, (int64_t)dim.z-1))));
              d = dists[idx];
              dvIt++;
              if(dvIt == deltaVec.end())
                break;

              
            }
          if(dvIt != deltaVec.end())
            outData[eidx] = outData[idx];
        }
    }

}
#endif

#endif //__RESAMPLE__
