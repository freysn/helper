#ifndef __POINTS2VOL__
#define __POINTS2VOL__

#ifdef USE_OMP
#include <omp.h>
#endif

#include "helper_idx.h"

template<typename F, typename F3>
  class Filter_Hat
{
 public:
 Filter_Hat() :
  mag(1.), exth(1.)
    {}
  F getWeight(const F3& v) const
  {
    F3 v2;
    v2.x = std::abs(v.x/exth);
    v2.y = std::abs(v.y/exth);
    v2.z = std::abs(v.z/exth);

    if(v2.x > 1. || v2.y > 1. || v2.z > 2.)
      return 0.;
    
    return
      mag*
      (1.-v2.x)*(1.-v2.y)*(1.-v2.z);
  }

  F operator ()(const F3& v) const
  {
    return getWeight(v);
  }
  
  F exth;
  F mag;
};


template<typename F3, typename F, typename I3>
  void splatWeightedPoints(std::vector<F>& vol,
                                     const std::vector<std::pair<F3, F>>& weightedPoints, const I3& volDim)
{
  vol.resize(volDim.x*volDim.y*volDim.z, 0.);
  
  Filter_Hat<float, F3> filter;
  filter.exth = 2.;
  filter.mag = 0.25;  
  
  for(size_t i=0; i<weightedPoints.size(); i++)
    {        
      const F3 p = weightedPoints[i].first;
      const double weight = weightedPoints[i].second;          

      V3<int> start,end;
      start.x = std::ceil(p.x-filter.exth);
      start.y = std::ceil(p.y-filter.exth);
      start.z = std::ceil(p.z-filter.exth);

      end.x = std::floor(p.x+filter.exth);
      end.y = std::floor(p.y+filter.exth);
      end.z = std::floor(p.z+filter.exth);

      start = iiiclamp(start, volDim);
      end = iiiclamp(end, volDim);
        
      assert(start.x <= end.x);
      assert(start.y <= end.y);          
      assert(start.z <= end.z);
        
      {
        V3<int> i;
        for(i.z=start.z; i.z<=end.z; i.z++)
          for(i.y=start.y; i.y<=end.y; i.y++)
            for(i.x=start.x; i.x<=end.x; i.x++)
              vol[iii2i(i, volDim)] +=
                //128.;
                weight*filter(make_vec<F3>(i)-p);
      }
    }     
}

template<typename F3, typename F, typename I3>
  std::vector<F> splatWeightedPoints(const std::vector<std::pair<F3, F>>& weightedPoints,
                                     const I3& volDim)
{
  std::vector<F> vol(volDim.x*volDim.y*volDim.z, 0.);
  splatWeightedPoints(vol,
                      weightedPoints,
                      volDim);
  return vol;
}

template<typename F3, typename F, typename I3>
  void 
  splatWeightedPointsTri(std::vector<F>& vol,
                         const std::vector<std::pair<F3, F>>& weightedPoints, const I3& volDim)
{
  const int nThreads = 
#ifdef USE_OMP_SPLAT_TRI
    omp_get_max_threads();
  #else
  1;
#endif

  const size_t nVoxels = volDim.x*volDim.y*volDim.z;
  vol.resize(nVoxels*nThreads, 0.);  

#ifdef USE_OMP_SPLAT_TRI
#pragma omp parallel for
#endif
  for(size_t i=0; i<weightedPoints.size(); i++)
    {
      const size_t threadId = 
#ifdef USE_OMP_SPLAT_TRI
        omp_get_thread_num();
#else
      0;      
#endif
      
      //const float3 p = weightedPoints[i].first;
      const F3 p = weightedPoints[i].first;
      const double weight = weightedPoints[i].second;

      
      if(weight==0)
        continue;

      assert(p.x >= 0. && p.y >= 0. && p.z >= 0.);


      //std::cout << "splat " << p.x << " " << p.y << " " << p.z << " " << weight << std::endl;
      V3<int> start,end;
      start.x = std::floor(p.x);
      start.y = std::floor(p.y);
      start.z = std::floor(p.z);

      end.x = start.x+1;
      end.y = start.y+1;
      end.z = start.z+1;

      start = iiiclamp(start, volDim);
      end = iiiclamp(end, volDim);
        
      assert(start.x <= end.x);
      assert(start.y <= end.y);          
      assert(start.z <= end.z);
        
      {
        V3<int> i;
        for(i.z=start.z; i.z<=end.z; i.z++)
          for(i.y=start.y; i.y<=end.y; i.y++)
            for(i.x=start.x; i.x<=end.x; i.x++)
              {
                
                double w3x = (1.-std::abs(p.x-i.x));
                double w3y = (1.-std::abs(p.y-i.y));
                double w3z = (1.-std::abs(p.z-i.z));
                
                w3x = std::max(0., std::min(w3x, 1.));
                w3y = std::max(0., std::min(w3y, 1.));
                w3z = std::max(0., std::min(w3z, 1.));
                
                const double w = w3x*w3y*w3z;
                assert(w >= 0.);//here
                assert(weight >= 0.);
              vol[iii2i(i, volDim)+threadId*nVoxels] +=
                w
                *
                weight;
              }
      }
    }
#ifdef USE_OMP_SPLAT_TRI
#pragma omp parallel for
  for(size_t i=0; i<nVoxels; i++)
    for(size_t j=1; j<nThreads; j++)
      vol[i] += vol[i+j*nVoxels];  
  vol.resize(nVoxels);
#endif
      
  
}

template<typename F3, typename F, typename I3>
  std::vector<F>
  splatWeightedPointsTri(const std::vector<std::pair<F3, F>>& weightedPoints, const I3& volDim)
{
  std::vector<F> vol;
  splatWeightedPointsTri(vol,
                         weightedPoints, volDim);
  return vol;
}

#endif //__POINTS2VOL__
