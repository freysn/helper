#ifndef __LOOKUP__
#define __LOOKUP__


/*
__device__ float4 helper_texLookup(float3 np)
{
  return tex1D(transferTex, tex3D(tex, np.x, np.y, np.z));
  //return tex1D(transferTex, evalFunc(pos.x, pos.y, pos.z));
  //return make_float4(evalFunc(pos.x, pos.y, pos.z));
}
*/

template<bool nearestMode, bool texNormalized, typename T, typename VOL>
#ifdef __CUDACC__
__device__
#endif
T normPos(T pos, VOL vol)
{  
  const T f3 = 
    vol.template remapNormalized<nearestMode, texNormalized>(pos);
  
  return f3;
}

template<bool nearestMode, bool texNormalized, typename T, typename F>
#ifdef __CUDACC__
__device__
#endif
  float4 fetchCol(float3 pos, T vol, F texLookup)
{
#if 1
  float3 np = vol.template remapNormalized<nearestMode, texNormalized>(pos);
#else
  float3 np = normPos<nearestMode, texNormalized>(pos, vol);
#endif
  return texLookup(np);
}
/*
template<bool nearestMode, bool texNormalized, typename T, typename I, typename F>
__device__ float4 fetchCol(float3 pos, T vol, I timeStep, F texLookup)
{
  float3 np = normPos<nearestMode, texNormalized>(pos, vol);
  np += vol.getElemCoordsAtlas(timeStep);
  return texLookup(np);
}
*/

template<typename T>
#ifdef __CUDACC__
__device__
#endif
float4 fetchCol(float posx, float posy, float posz, T texLookup)
{
  return texLookup(make_float3(posx, posy, posz));
}

#endif //__LOOKUP__
