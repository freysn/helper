#ifndef __LOOKUP__
#define __LOOKUP__


/*
__device__ V4<float> helper_texLookup(V3<float> np)
{
  return tex1D(transferTex, tex3D(tex, np.x, np.y, np.z));
  //return tex1D(transferTex, evalFunc(pos.x, pos.y, pos.z));
  //return make_V4<float>(evalFunc(pos.x, pos.y, pos.z));
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

template<bool nearestMode, bool texNormalized, typename T, typename F, typename POS>
#ifdef __CUDACC__
__device__
#endif
auto fetchCol(const POS& pos, const T& vol, F texLookup)
{
#if 1
  const auto np = vol.template remapNormalized<nearestMode, texNormalized>(pos);
#else
  V3<float> np = normPos<nearestMode, texNormalized>(pos, vol);
#endif
  return texLookup(np);
}
/*
template<bool nearestMode, bool texNormalized, typename T, typename I, typename F>
__device__ V4<float> fetchCol(V3<float> pos, T vol, I timeStep, F texLookup)
{
  V3<float> np = normPos<nearestMode, texNormalized>(pos, vol);
  np += vol.getElemCoordsAtlas(timeStep);
  return texLookup(np);
}
*/

template<typename T>
#ifdef __CUDACC__
__device__
#endif
V4<float> fetchCol(float posx, float posy, float posz, T texLookup)
{
  return texLookup(V3<float>(posx, posy, posz));
}

#endif //__LOOKUP__
