#ifndef __OVER__
#define __OVER__

#ifndef __CUDACC__
#include <cmath>
#endif

// "over" operator for front-to-back blending
template<typename S, typename T, typename W>
#ifdef __CUDACC__
__host__ __device__
#endif
void over(S& sum, const T& col, const W& w)
{
  S tmp;
  tmp.x = w*col.x;
  tmp.y = w*col.y;
  tmp.z = w*col.z;
  tmp.w = w;
  
  //sum = sum + make_float4(w*col.x, w*col.y, w*col.z, w)*(1.0f - sum.w);
  sum = sum + tmp*((W)1.0 - sum.w);
}

template<typename S, typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
void over(S& sum, const T& col)
{
  over(sum, col, col.w);
}

template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T adjustOpacityContribution(T w, T tstepModifier)
{
  //#ifdef __CUDACC__
#ifdef __CUDA_ARCH__
  return 1.f - __powf(1.f-w, tstepModifier);    
#else
  return 1. - std::pow(1.-w, tstepModifier);
#endif
    
}

#endif //__OVER__
