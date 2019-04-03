#ifndef __LIGHTS__
#define __LIGHTS__

#if 1
#include "m_vec.h"

#warning "GETLIGHTSDEFAULT IS DEPRECATED, BUT STILL USED TO CREATE SHADOW TEXTURE"
typedef m_vec<float3, float3, float3, float3> lights_vec;
#ifdef __NVCC__
__device__
#endif
lights_vec getLightsDefault()
//m_vec<float3, float3> getLightsDefault()
{
  return lights_vec
    (//make_float3(7.f, 7.f, 7.f),
     make_float3(0.f, -7.f, -7.f),
     make_float3(-7.f, 0.f, 7.f),
     make_float3(7.f, 0.f, -7.f),
     make_float3(0.f, 7.f, 0.f)
     //make_float3(3.f, 7.f, 0.f)
     );
}
#endif

#endif //__LIGHTS__