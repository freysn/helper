#ifndef __RAY_SETUP__
#define __RAY_SETUP__

#include "grd/PCam.h"

namespace helper
{
template<typename F2, typename F3, typename F4>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  F2 getScreenPos(const PCam<F3, F4>& pcam, double x, double y)
{
  double u = (x / (double) pcam.imageW)*2.0f-1.0f;
  double v = (y / (double) pcam.imageH)*2.0f-1.0f;
  return make_vec<F2>(u,v);
};

template<typename F, typename F3, typename F4>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto getScreenPos_inv(const PCam<F3, F4>& pcam, F u, F v)
{
  V2<F> out;
  out.x = ((u+1.)/2.)*pcam.imageW;
  out.y = ((v+1.)/2.)*pcam.imageH;
  return out;
};


template<typename F, typename F3, typename F4>
#ifdef __CUDACC__
__host__ __device__
#endif
F3 getRayD_nonorm_uv(const PCam<F3, F4>& pcam, F u, F v)
{
  auto dir = make_vec<F3>(pcam.center+
			 u*pcam.apertureRatioX*pcam.right+
			 v*pcam.apertureRatioY*pcam.up);

#ifndef __CUDACC__
 assert(length(dir) > 0.);
#endif
 return dir;
}

  template</*typename T,*/ typename F3, typename F4, typename F3B>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto getRayD_nonorm_uv_inv(const PCam<F3, F4>& pcam, /*V3<T>*/F3B dir)
{
  using T = decltype(dir.x);
  
  const auto center = V3<T>(pcam.center.x, pcam.center.y, pcam.center.z);
  
  const auto d = V3<T>(dir.x, dir.y, dir.z) - center;

  assert(std::abs(dot(d, center)) < 0.001);

  V2<T> uv;


  uv.x = dot(V3<T>(pcam.right.x, pcam.right.y, pcam.right.z), d)/pcam.apertureRatioX;
  uv.y = dot(V3<T>(pcam.up.x, pcam.up.y, pcam.up.z), d)/pcam.apertureRatioY;

  return uv;
}

template<typename F3, typename F4>
  #ifdef __CUDACC__
  __host__ __device__
#endif  
F3 getRayD_nonorm(const PCam<F3, F4>& pcam, double x, double y)
{          
  auto screenPos = getScreenPos<V2<double>>(pcam, x,y);
#ifndef NDEBUG
  {
    auto xy = getScreenPos_inv(pcam, screenPos.x, screenPos.y);
    assert(xy.eq(decltype(xy)(x,y), 0.1));
  }
#endif

  auto dir = getRayD_nonorm_uv(pcam, screenPos.x, screenPos.y);

#ifndef NDEBUG
  {
    auto uv = getRayD_nonorm_uv_inv(pcam, dir);
    assert(uv.eq(convert<typename decltype(uv)::value_type>(screenPos), 0.01));
  }
#endif

  return dir;
}
 
template<typename F3, typename F4>
  #ifdef __CUDACC__
  __host__ __device__
#endif  
  F3 getRayD(const PCam<F3, F4>& pcam, double x, double y)
{
  return normalize(getRayD_nonorm(pcam, x, y));
};

}
#endif //__RAY_SETUP__
