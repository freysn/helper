#ifndef __LIGHTS__
#define __LIGHTS__

#if 1
#include "m_vec.h"

#warning "GETLIGHTSDEFAULT IS DEPRECATED, BUT STILL USED TO CREATE SHADOW TEXTURE"
typedef m_vec<V3<float>, V3<float>, V3<float>, V3<float>> lights_vec;
#ifdef __NVCC__
__device__
#endif
lights_vec getLightsDefault()
//m_vec<V3<float>, V3<float>> getLightsDefault()
{
  return lights_vec
    (//V3<float>(7.f, 7.f, 7.f),
     V3<float>(0.f, -7.f, -7.f),
     V3<float>(-7.f, 0.f, 7.f),
     V3<float>(7.f, 0.f, -7.f),
     V3<float>(0.f, 7.f, 0.f)
     //V3<float>(3.f, 7.f, 0.f)
     );
}
#endif

#endif //__LIGHTS__