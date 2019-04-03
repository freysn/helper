#ifndef __GRADIENT__
#define __GRADIENT__


template<bool texNormalized, typename T, typename F>
__device__ float3 gradientWithCentralDifferences(float3 pos, float delta, T vol, F texLookup)
{  
  float3 diff =  
    make_float3(
		fetchCol<false, texNormalized>
		(make_float3(pos.x - delta, pos.y, pos.z), vol, texLookup).w-
		fetchCol<false, texNormalized>
		(make_float3(pos.x + delta, pos.y, pos.z), vol, texLookup).w,
		fetchCol<false, texNormalized>
		(make_float3(pos.x, pos.y - delta, pos.z), vol, texLookup).w -
		fetchCol<false, texNormalized>
		(make_float3(pos.x, pos.y + delta, pos.z), vol, texLookup).w,
		fetchCol<false, texNormalized>
		(make_float3(pos.x, pos.y, pos.z - delta), vol, texLookup).w -
		fetchCol<false, texNormalized>
		(make_float3(pos.x, pos.y, pos.z + delta), vol, texLookup).w);
  
  //return diff/(2.f*delta);
  return diff;
}



template<bool texNormalized, typename T, typename F>
__device__ float3 gradientWithSobel3D(float3 pos, float delta, T vol, F texLookup)
{
  float3 gradient;
  
  //
  // X
  //
  gradient.x = 
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y, pos.z), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w);


  gradient.x += 
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y, pos.z), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);


  //
  // Y
  //
  gradient.y = 
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y-delta, pos.z), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w);

  gradient.y += 
    (fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y+delta, pos.z), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);
  
  //
  // Z
  //
  gradient.z = 
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-6.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y, pos.z-delta), vol, texLookup).w) +
    (-3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-3.*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y+delta, pos.z-delta), vol, texLookup).w) +
    (-fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z-delta), vol, texLookup).w);

  gradient.z += 
    (fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y-delta, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (6.f*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y, pos.z+delta), vol, texLookup).w) +
    (3.f*fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x-delta, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (3.*fetchCol<false, texNormalized>(make_float3(pos.x, pos.y+delta, pos.z+delta), vol, texLookup).w) +
    (fetchCol<false, texNormalized>(make_float3(pos.x+delta, pos.y+delta, pos.z+delta), vol, texLookup).w);

  return gradient/22.;
}

template<bool texNormalized, typename T, typename F>
__device__ float3 gradientDefault(float3 pos, float delta, T vol, F texLookup)
{
  return gradientWithCentralDifferences<texNormalized>(pos, delta, vol, texLookup);
  //return gradientWithSobel3D<texNormalized>(pos, delta, vol, texLookup);
}


#endif //__GRADIENT__