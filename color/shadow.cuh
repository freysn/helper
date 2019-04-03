#ifndef __SHADOW__
#define __SHADOW__

#include "m_vec.h"

template<bool nearestMode, bool texNormalized, typename T, typename F>
__device__
float traceShadowRay(const float3 lightPos, const float3 pos, const T vol, const F& texLookup, const float tstepModifierShadow, const float tstepOff)
{
  Ray eyeRay;
  eyeRay.o = pos;
  eyeRay.d = normalize(lightPos-pos);

  float tnear, tfar;
  int hit = intersectBox(eyeRay, vol.boxMin, vol.boxMax, &tnear, &tfar);
  
  if (!hit)
    return 0.f;
  
  if (tnear < 0.0f)
    tnear = 0.0f;
  
  float op = 0.;
  const float tstep = vol.tstepRef*tstepModifierShadow;
  
  float t = tnear+tstepOff*vol.tstepRef;

  //printf("%f %f\n", tnear, tfar);
  while(op<0.99 && t < tfar)
    {
      const float3 p = eyeRay.o+t*eyeRay.d;
      
      const float w = adjustOpacityContribution(fetchCol<nearestMode, texNormalized>
						(p, vol, texLookup).w,
						tstepModifierShadow);
      

      op += w*(1.f-op);
      t += tstep;      
    }

  //op = 0.7f;
  return op;
}

template<bool nearestMode, bool texNormalized, typename O, typename T, typename F>
struct _traceShadowRays_ShadowOp
  {
    template<size_t k>
    __device__
    void operator()(const float3& lightPos)
    {
      const float shadow =
	traceShadowRay<nearestMode, texNormalized>
	(lightPos, pos, vol, texLookup, tstepModifier, tstepOff);
      m_assign<k>(ops, shadow);
    }

    float3 pos;
    T vol;
    F texLookup;
    float tstepModifier;
    float tstepOff;
    O ops;
  };

template<bool nearestMode, bool texNormalized, typename O, typename L, typename T, typename F>
__device__
void traceShadowRays(O& ops, L lights, float3 pos, T vol, F texLookup, float tstepModifier, const float tstepOff)
{
  _traceShadowRays_ShadowOp<nearestMode, texNormalized, O, T, F> shadowOp;

  shadowOp.pos = pos;
  shadowOp.vol = vol;
  shadowOp.texLookup = texLookup;
  shadowOp.tstepModifier = tstepModifier;
  shadowOp.tstepOff = tstepOff;
  
  m_forall(shadowOp, lights);

  ops = shadowOp.ops;
}
  
#endif //__SHADOW__