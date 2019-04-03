#ifndef __VOL_IDX_HELPER__
#define __VOL_IDX_HELPER__

#include <list>
#include <cstddef>
#include <cassert>


template<typename I3A, typename I3B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iii2i(const I3A& v, const I3B& volDim)
{
  typedef size_t I;
  return (I)v.x+volDim.x*((I)v.y+volDim.y*(I)v.z);
}
#if 0
template<typename I3>
  size_t iii2n(const I3& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y*(I)volDim.z;
}

template<>
  size_t iii2i(const std::pair<unsigned int, std::list<unsigned int> >& v, const int3& volDim)
{
  return v.first;
}
#endif

template<typename I3>
  size_t iii2n(const I3& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y*(I)volDim.z;
}

/*
  uint32_t iii2i(const uint32_t& v, const int3& volDim)
  {
    return v;
  }
*/

  template<typename I, typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  I3 i2iii(I v, const I3& volDim)
{
  I3 out;
  out.z = v/(volDim.x*volDim.y);
  v -= out.z*volDim.x*volDim.y;
  out.y = v/volDim.x;
  out.x = v-out.y*volDim.x;

  //std::cout << __func__ << " " << v << " " << volDim.x << " " << volDim.y << " " << volDim.z << std::endl;
  assert(out.x < volDim.x
	 && out.y < volDim.y
	 && out.z < volDim.z);
  return out;
}
  
  template<typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  I3 iiiclamp(const I3& v, const I3& volDim)
  {
        #ifndef __CUDACC__
    using namespace std;
#endif  
    I3 o;
    o.x = max(static_cast<decltype(v.x)>(0), min(v.x, volDim.x-1));
    o.y = max(static_cast<decltype(v.y)>(0), min(v.y, volDim.y-1));
    o.z = max(static_cast<decltype(v.z)>(0), min(v.z, volDim.z-1));
    return o;
  }

template<typename I3>
bool isWithinBounds(const I3& v, const I3& volDim)
{
  return
    v.x >= 0 && v.x < volDim.x
    && v.y >= 0 && v.y < volDim.y
    && v.z >= 0 && v.z < volDim.z;
}

template<typename I3>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iii2i_clamp(const I3& v, const I3& volDim)
  {
    return iii2i(iiiclamp(v, volDim), volDim);
  }

#endif //__VOL_IDX_HELPER__
