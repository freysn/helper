#ifndef __LOCAL_ILLUM__
#define __LOCAL_ILLUM__

//#include "../grd/PVol.h"
//#include "../grd/PVolAtlas.h"
#include "over.h"
#include "lookup.h"
#include "m_vec.h"
//#include "shadow.cuh"
#include "lights.cuh"
#include "gradient.cuh"

#ifndef __CUDACC__
template<typename T>

T saturate(const T& x)
{
  if(x<0)
    return 0;
  else if(x>1)
    return 1;
  else
    return x;
}
#endif


V3<float> blinnPhongShading(V3<float> ambientColor, V3<float> lightPos, V3<float> lightDiffuseColor, float lightDiffusePower, V3<float> lightSpecularColor, float lightSpecularPower, V3<float> pos3D, V3<float> viewDir, V3<float> normal)
{
  //V3<float> ambient = ambientColor;
  V3<float> diffuse(0., 0., 0.);
  V3<float> specular(0., 0., 0.);
    
  if(lightDiffusePower > 0)
    {
      V3<float> lightDir = lightPos - pos3D; // FIND THE VECTOR BETWEEN THE 3D POSITION IN SPACE OF THE SURFACE
      float distance = length(lightDir); // GET THE DISTANCE OF THIS VECTOR
      distance = distance * distance; // USES INVERSE SQUARE FOR DISTANCE ATTENUATION
      distance = 1.f;
      lightDir = normalize(lightDir); // NORMALIZE THE VECTOR

      // INTENSITY OF THE DIFFUSE LIGHT
      // SATURATE TO KEEP WITHIN THE 0-1 RANGE
      // DOT PRODUCT OF THE LIGHT DIRECTION VECTOR AND THE SURFACE NORMAL
      float i = saturate(dot(lightDir, normal));
      //float i = fabs(dot(lightDir, normal));
      //float i = 1.f;

      diffuse = i * lightDiffuseColor * lightDiffusePower / distance; // CALCULATE THE DIFFUSE LIGHT FACTORING IN LIGHT COLOUR, POWER AND THE ATTENUATION

      //CALCULATE THE HALF VECTOR BETWEEN THE LIGHT VECTOR AND THE VIEW VECTOR. THIS IS CHEAPER THAN CALCULATING THE ACTUAL REFLECTIVE VECTOR
      V3<float> h = normalize(lightDir + viewDir);

	  const float specularHardness = 4.f;
      // INTENSITY OF THE SPECULAR LIGHT
      // DOT PRODUCT OF NORMAL VECTOR AND THE HALF VECTOR TO THE POWER OF THE SPECULAR HARDNESS
      i = pow(saturate(dot(normal, h)), specularHardness);

      specular = i * lightSpecularColor * lightSpecularPower / distance; // CALCULATE THE SPECULAR LIGHT FACTORING IN LIGHT SPECULAR COLOUR, POWER AND THE ATTENUATION
    }
  //return 0.5*ambient + 0.6*diffuse + 1. * specular;
  //return 1.f*diffuse * 0.5f*specular;
  return diffuse+0.3f*specular;
}


void simpleCompositing(V4<float>& sum, float& w, V4<float>& col, float tstepModifier)
{  
  w = adjustOpacityContribution(col.w, tstepModifier);
  over(sum, col, w);  
}

struct ColOp
  {
    template<size_t k>
    
    void operator()(const V3<float>& lightPos)
    {
      float shadow = m_get<k>(shadows);

#if SHADOW_MODE>0
      if(shadow < 0.95)
#endif
	{
	  /*
#if SHADOW_MODE==0
	  const float lightContribFac = 0.3;	  
#else
	  //const float lightContribFac = 0.6;
	  const float lightContribFac = 0.1;
#endif
	  */

	  float dotLightGradient = 0.25;

	  if(gradientLen > 0.0001)
	    {
	      const V3<float> lightDirNorm = normalize(lightPos-pos);
	      //const V3<float> lightDirNorm = V3<float>(0., 0., 1.);;

	      dotLightGradient = dot(lightDirNorm, gradientNorm);
	      //dotLightGradient *= 0.8;
	      if(dotLightGradient<0.)
		return;
	      //dotLightGradient = min(dotLightGradient, 0.5);
	    }
	  
	    /*
	  const float fac = lightContribFac*(1.f-shadow);
	  const V3<float> specularColor = V3<float>(1.f);
	  const V3<float> diffspec = blinnPhongShading(V3<float>(0.f), lightPos, diffuseColor, 1.f, specularColor, 1.f, pos, rayDir, gradientNorm);
*/
	  //colRGB += diffuseColor;

	  const V3<float> shadedCol = dotLightGradient*diffuseColor;
	  
	  //colRGB += V3<float>(0.3*tmp, 0., tmp);


	  
	  //const float w = min(gradientLen*3.f, 1.f);
	  //const float w = 1.;

	  colRGB += shadedCol*(1.-shadow);
	  //colRGB += shadedCol;
	  
	  //const float w = 0.f;
	  /*
	  colRGB += fac*(w*diffspec + 0.7f*(1.f-w)*diffuseColor);
	  //colRGB += fac*(w*diffspec + (1.f-w)*diffuseColor);
	  */
	}
    }
    
    V3<float> colRGB;// = V3<float>(0.f);
    V3<float> gradientNorm;
    float gradientLen;
    V3<float> diffuseColor;
    //V3<float> specularColor = V3<float>(1.f);
    V3<float> pos;
    V3<float> rayDir;
    V4<float> shadows;    
  };

template<typename L>

V3<float> lighting(L lights, V3<float> col, V3<float> rayDir, V3<float> pos, V3<float> gradient, V4<float> shadows/*, float opacity*/)
{
  /*
  //return col;
  if(length(gradient)<0.1)
    return V3<float>(1., 0., 0.);
  return V3<float>(dot(normalize(lights.tail-pos), normalize(gradient)),
		     dot(normalize(lights.tail-pos), normalize(gradient)),
		     length(gradient));
  //auto lights = getLightsDefault();  
  */
  ColOp colOp;

  colOp.gradientLen = length(gradient);
  colOp.gradientNorm = gradient/colOp.gradientLen;
  colOp.diffuseColor = col;
  colOp.pos = pos;
  colOp.rayDir = rayDir;
  colOp.shadows = shadows;

  //ambient term
  //colOp.colRGB = 0.25f*col;
  //colOp.colRGB = 0.35f*col;

  
  //colOp.colRGB = 0.25*max(1.-colOp.gradientLen, 0.)*col;
  //colOp.colRGB = col;

  const uint8_t mode=1;
  
  if(mode)
    colOp.colRGB = 0.55*col;
  else
    //used this for bottle
    colOp.colRGB = 0.6*col;

  
  //if(colOp.gradientLen > 0.0001)
  m_forall(colOp, lights);

  //const float gradientScale = 4.*colOp.gradientLen;
  if(mode)
    colOp.colRGB *= 0.45;
  else
    colOp.colRGB *= 0.7;

#ifdef SHADOW_MODE
#if SHADOW_MODE==0
  colOp.colRGB *= 0.85;
#endif
#endif

  //colOp.colRGB += col*(1.-opacity)*0.7*max(1.-gradientScale, 0.);

  //colOp.colRGB = colOp.diffuseColor;
  return colOp.colRGB;  
}

#if 0
template<typename T, typename F>
struct _lighting_ColOp
  {
    template<size_t k>
    
    void operator()(const V3<float>& lightPos)
    {
      const float shadow = traceShadowRay(lightPos, pos, vol, texLookup, 1.f);

      if(shadow < 0.95)
	{
	  const float fac = 1.3*(1.f-shadow);
	  const V3<float> specularColor = V3<float>(1.f);
	  colRGB += fac*blinnPhongShading(V3<float>(0.f), lightPos, diffuseColor, 1.f, specularColor, 1.f, pos, rayDir, gradientNorm);
	}
    }
    
    V3<float> colRGB = V3<float>(0.f);
    V3<float> gradientNorm;
    V3<float> diffuseColor;
    V3<float> specularColor = V3<float>(1.f);
    V3<float> pos;
    V3<float> rayDir;
    T vol;
    F texLookup;
  };

template<bool texNormalized, typename T, typename F>

V3<float> lighting(V3<float> col, V3<float> rayDir, V3<float> pos, T vol,
		float tstepModifier, float deltaGradient, F texLookup, int timeStep)
{
  auto lights = getLightsDefault();
  
  /* const V3<float> lightPos3 = V3<float>(2.f, -2.f, -2.f); */
  /* const V3<float> lightPos4 = V3<float>(-2.f, 2.f, 2.f); */
  /* const V3<float> lightPos5 = V3<float>(-2.f, 2.f, -2.f); */
  /* const V3<float> lightPos6 = V3<float>(-2.f, -2.f, 2.f); */
  /* const V3<float> lightPos7 = V3<float>(-2.f, -2.f, -2.f); */
      
      
      
  V3<float> ambientColor = col;
  
  //V3<float> gradient = gradientWithSobel3D(pos, deltaGradient);
  ////V3<float> gradient = gradientWithSobel3D(pos, 4*deltaGradient, vol);
  const V3<float> gradient = gradientWithCentralDifferences<texNormalized>(pos, deltaGradient, vol, texLookup);
  const float gradientLen = length(gradient);  

  _lighting_ColOp<T,F> colOp;

  colOp.vol = vol;
  colOp.texLookup = texLookup;
  colOp.diffuseColor = col;
  colOp.gradientNorm = gradient/gradientLen;
  colOp.pos = pos;
  colOp.rayDir = rayDir;

  m_forall(colOp, lights);
  
  //if(length(gradient) > 0.f)
  {    
    //const float gradientLenMax = 0.3f;
    const float gradientLenMax = 1.f;
    float weightDiffSpec = min(gradientLen/gradientLenMax,1.f);
    float weightAmbient = 1.f-weightDiffSpec+.2f;
    
    colOp.colRGB = weightDiffSpec*colOp.colRGB+weightAmbient*ambientColor;
  }  
            
  return colOp.colRGB;
  //#endif
}


template<bool texNormalized, typename T, typename F>

void lightingCompositing(V4<float>& sum, float& w, V4<float>& col, V3<float> rayDir, V3<float> pos, T vol, float tstepModifier, float deltaGradient, F texLookup, int timeStep=0)
{              
  //w = 1.f - __powf(1.f-col.w, tstepModifier);      
  ////w = adjustOpacityContribution(col.w, tstepModifier);
          
  //if(col.w > 0.0001f)
  {
    const V3<float> colRGB =
      lighting<texNormalized>(V3<float>(col), rayDir, pos, vol,
			      tstepModifier,
			      deltaGradient, texLookup, timeStep);
    col.x = colRGB.x;
    col.y = colRGB.y;
    col.z = colRGB.z;
  }
        
      
  ////over(sum, col, w);

  simpleCompositing(sum, w, col, tstepModifier);
}
#endif

#endif //__LOCAL_ILLUM__
