#ifndef __SHADOW__
#define __SHADOW__

#include "m_vec.h"
#include "helper/color/over.h"
#include "helper/color/lookup.h"

template<bool nearestMode, bool texNormalized, typename T, typename F>

double traceShadowRay(const V3<double> lightPos, const V3<double> pos, const T vol, const F& texLookup, const double tstepModifierShadow, const double tstepOff)
{
  Ray eyeRay;
  eyeRay.o = pos;

  //eyeRay.d = normalize(lightPos-pos);
  eyeRay.d=lightPos-pos;
  const auto len=length(eyeRay.d);
  
  eyeRay.d /= len;

  double tnear, tfar;
  int hit = helper::intersectBox(eyeRay, vol.boxMin, vol.boxMax, &tnear, &tfar);
  
  if (!hit)
    return 0.f;
  
  if (tnear < 0.0f)
    tnear = 0.0f;

  // if light source is within volume
  if(len < tfar)
    tfar=len;
  
  double op = 0.;
  const double tstep = vol.tstepRef*tstepModifierShadow;
  
  double t = tnear+tstepOff*vol.tstepRef;

  //printf("%f %f\n", tnear, tfar);
  while(op<0.99 && t < tfar)
    {
      const V3<double> p = eyeRay.o+t*eyeRay.d;

      const auto c = fetchCol<nearestMode, texNormalized>(p, vol, texLookup);
      const double w = adjustOpacityContribution(c.w, tstepModifierShadow);
      

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
    
    void operator()(const V3<double>& lightPos)
    {
      const double shadow =
	traceShadowRay<nearestMode, texNormalized>
	(lightPos, pos, vol, texLookup, tstepModifier, tstepOff);
      m_assign<k>(ops, shadow);
    }

    V3<double> pos;
    T vol;
    F texLookup;
    double tstepModifier;
    double tstepOff;
    O ops;
  };

template<bool nearestMode, bool texNormalized, typename O, typename L, typename T, typename F>

void traceShadowRays(O& ops, L lights, V3<double> pos, T vol, F texLookup, double tstepModifier, const double tstepOff)
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