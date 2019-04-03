#ifndef __DOWNSAMPLE__
#define __DOWNSAMPLE__

#include <vector>
//#include <cstdint>
#include <limits>
#include <cmath>

template<typename TM, typename T, typename I3>
  void reduceNearest(std::vector<T>& outData, I3& outDim, T* inData, I3 inDim, int reductionFactor)
{
  outDim.x = inDim.x/reductionFactor;
  outDim.y = inDim.y/reductionFactor;
  outDim.z = inDim.z/reductionFactor;

  outData.resize(outDim.x*outDim.y*outDim.z, 0);
  
  for(int64_t z=0; z<outDim.z; z++)
    for(int64_t y=0; y<outDim.y; y++)
      for(int64_t x=0; x<outDim.x; x++)
        {
          TM v = 0;
          for(int64_t zi=z*reductionFactor; zi<(z+1)*reductionFactor; zi++)
            for(int64_t yi=y*reductionFactor; yi<(y+1)*reductionFactor; yi++)
              for(int64_t xi=x*reductionFactor; xi<(x+1)*reductionFactor; xi++)
                v += inData[xi+inDim.x*(yi+inDim.y*zi)];
          
          v /= (reductionFactor*reductionFactor*reductionFactor); 
          outData[x+outDim.x*(y+outDim.y*z)] = 
            std::max((TM)std::numeric_limits<T>::min(),
                     std::min(v, (TM)std::numeric_limits<T>::max()));
        }
}


template<typename TM, typename T, typename I3>
  void reduceNearest(std::vector<T>& outData, I3& outDim, const std::vector<T>& inData, I3 inDim, int reductionFactor)
{
  reduceNearest<TM>(outData, outDim, (T*)&(inData[0]), inDim, reductionFactor);
}

template<typename I3>
I3 dimFromReductionFactor(const I3& inDim, const int reductionFactor)
{
  I3 outDim;
  outDim.x = std::max(inDim.x/reductionFactor, 1);
  outDim.y = std::max(inDim.y/reductionFactor, 1);
  outDim.z = std::max(inDim.z/reductionFactor, 1);
  return outDim;
}


  
template<typename TM, typename T, typename I3>
  void reduceLinear(std::vector<T>& outData, I3& outDim,
                    const std::vector<T>& inData, const I3 inDim,
                    const int reductionFactor)
{
  outDim = dimFromReductionFactor(inDim, reductionFactor);
  
  outData.resize(outDim.x*outDim.y*outDim.z);
  std::fill(outData.begin(), outData.end(), 0);

  std::vector<std::pair<I3, TM> > deltaVec;

  {
    I3 v;
    TM sumw = 0.;
    TM reductionFactor_tm = reductionFactor;
    for(v.z=-reductionFactor+1; v.z<reductionFactor; v.z++)
      for(v.y=-reductionFactor+1; v.y<reductionFactor; v.y++)
        for(v.x=-reductionFactor+1; v.x<reductionFactor; v.x++)
          {
            const TM w =
              ((1.-std::abs(v.x)/reductionFactor_tm)
               *(1.-std::abs(v.y)/reductionFactor_tm)
               *(1.-std::abs(v.z)/reductionFactor_tm));

            deltaVec.emplace_back(v,w);
            sumw += w;
          }
    for(auto& d : deltaVec)
      d.second /= sumw;
  }
  
  for(int64_t oz=0; oz<outDim.z; oz++)
    for(int64_t oy=0; oy<outDim.y; oy++)
      for(int64_t ox=0; ox<outDim.x; ox++)
        {
          TM v=0.;
          int64_t ix = ox*reductionFactor+reductionFactor/2;
          int64_t iy = oy*reductionFactor+reductionFactor/2;
          int64_t iz = oz*reductionFactor+reductionFactor/2;
          
          for(auto d: deltaVec)
            {
              int64_t px = std::max((int64_t)0, std::min((ix+d.first.x),
                                                         (int64_t) inDim.x-1));
              int64_t py = std::max((int64_t)0, std::min((iy+d.first.y),
                                                         (int64_t) inDim.y-1));
              int64_t pz = std::max((int64_t)0, std::min((iz+d.first.z),
                                                         (int64_t) inDim.z-1));
              
              TM s = inData[px+inDim.x*(py+inDim.y*pz)];
              //std::cout << px << " " << py << " " << pz << " " << s << " " << d.second << std::endl;
              v += s*d.second;
            }
          
          outData[ox+outDim.x*(oy+outDim.y*oz)] = 
            std::max((TM)std::numeric_limits<T>::min(),
                     std::min(v, (TM)std::numeric_limits<T>::max()));
        }
}


template<typename TM, typename T, typename I3>
  void reduceLinear_inPlace(std::vector<T>& data, I3& outDim,
                            const I3 inDim,
                            const int reductionFactor)
{
  const std::vector<T> dataIn = data;
  data.clear();
  reduceLinear<TM>(data, outDim, dataIn, inDim, reductionFactor);
}

#endif //__DOWNSAMPLE__
