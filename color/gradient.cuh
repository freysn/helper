#ifndef __GRADIENT__
#define __GRADIENT__

#include "helper/color/lookup.h"

template<bool texNormalized, typename T, typename F>
 V3<float> gradientWithCentralDifferences(V3<float> pos, float delta, T vol, F texLookup)
{  
  V3<float> diff =  
    V3<float>(
		fetchCol<false, texNormalized>
		(V3<float>(pos.x - delta, pos.y, pos.z), vol, texLookup).w-
		fetchCol<false, texNormalized>
		(V3<float>(pos.x + delta, pos.y, pos.z), vol, texLookup).w,
		fetchCol<false, texNormalized>
		(V3<float>(pos.x, pos.y - delta, pos.z), vol, texLookup).w -
		fetchCol<false, texNormalized>
		(V3<float>(pos.x, pos.y + delta, pos.z), vol, texLookup).w,
		fetchCol<false, texNormalized>
		(V3<float>(pos.x, pos.y, pos.z - delta), vol, texLookup).w -
		fetchCol<false, texNormalized>
		(V3<float>(pos.x, pos.y, pos.z + delta), vol, texLookup).w);
  
  //return diff/(2.f*delta);
  return diff;
}



template<bool texNormalized, typename T, typename F>
 V3<float> gradientWithSobel3D(V3<float> pos, float delta, T vol, F texLookup)
{
  V3<float> gradient;
  
  //
  // X
  //
  gradient.x = 
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y, pos.z), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w);


  gradient.x += 
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y, pos.z), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);


  //
  // Y
  //
  gradient.y = 
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y-delta, pos.z), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w);

  gradient.y += 
    (fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y+delta, pos.z), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);
  
  //
  // Z
  //
  gradient.z = 
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w);

  gradient.z += 
    (fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(V3<float>(pos.x, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(V3<float>(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);

  return gradient/22.;
}

template<bool texNormalized, typename T, typename F>
 V3<float> gradientDefault(V3<float> pos, float delta, T vol, F texLookup)
{
  return gradientWithCentralDifferences<texNormalized>(pos, delta, vol, texLookup);
  //return gradientWithSobel3D<texNormalized>(pos, delta, vol, texLookup);
}


#endif //__GRADIENT__