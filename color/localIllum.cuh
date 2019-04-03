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
__device__
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

__device__
float3 blinnPhongShading(float3 ambientColor, float3 lightPos, float3 lightDiffuseColor, float lightDiffusePower, float3 lightSpecularColor, float lightSpecularPower, float3 pos3D, float3 viewDir, float3 normal)
{
  //float3 ambient = ambientColor;
  float3 diffuse = make_float3(0.f);
  float3 specular = make_float3(0.f);
    
  if(lightDiffusePower > 0)
    {
      float3 lightDir = lightPos - pos3D; // FIND THE VECTOR BETWEEN THE 3D POSITION IN SPACE OF THE SURFACE
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
      float3 h = normalize(lightDir + viewDir);

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

__device__
void simpleCompositing(float4& sum, float& w, float4& col, float tstepModifier)
{  
  w = adjustOpacityContribution(col.w, tstepModifier);
  over(sum, col, w);  
}

struct ColOp
  {
    template<size_t k>
    __device__
    void operator()(const float3& lightPos)
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
	      const float3 lightDirNorm = normalize(lightPos-pos);
	      //const float3 lightDirNorm = make_float3(0., 0., 1.);;

	      dotLightGradient = dot(lightDirNorm, gradientNorm);
	      //dotLightGradient *= 0.8;
	      if(dotLightGradient<0.)
		return;
	      //dotLightGradient = min(dotLightGradient, 0.5);
	    }
	  
	    /*
	  const float fac = lightContribFac*(1.f-shadow);
	  const float3 specularColor = make_float3(1.f);
	  const float3 diffspec = blinnPhongShading(make_float3(0.f), lightPos, diffuseColor, 1.f, specularColor, 1.f, pos, rayDir, gradientNorm);
*/
	  //colRGB += diffuseColor;

	  const float3 shadedCol = dotLightGradient*diffuseColor;
	  
	  //colRGB += make_float3(0.3*tmp, 0., tmp);


	  
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
    
    float3 colRGB;// = make_float3(0.f);
    float3 gradientNorm;
    float gradientLen;
    float3 diffuseColor;
    //float3 specularColor = make_float3(1.f);
    float3 pos;
    float3 rayDir;
    float4 shadows;    
  };

template<typename L>
__device__
float3 lighting(L lights, float3 col, float3 rayDir, float3 pos, float3 gradient, float4 shadows, float opacity)
{
  /*
  //return col;
  if(length(gradient)<0.1)
    return make_float3(1., 0., 0.);
  return make_float3(dot(normalize(lights.tail-pos), normalize(gradient)),
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
    __device__
    void operator()(const float3& lightPos)
    {
      const float shadow = traceShadowRay(lightPos, pos, vol, texLookup, 1.f);

      if(shadow < 0.95)
	{
	  const float fac = 1.3*(1.f-shadow);
	  const float3 specularColor = make_float3(1.f);
	  colRGB += fac*blinnPhongShading(make_float3(0.f), lightPos, diffuseColor, 1.f, specularColor, 1.f, pos, rayDir, gradientNorm);
	}
    }
    
    float3 colRGB = make_float3(0.f);
    float3 gradientNorm;
    float3 diffuseColor;
    float3 specularColor = make_float3(1.f);
    float3 pos;
    float3 rayDir;
    T vol;
    F texLookup;
  };

template<bool texNormalized, typename T, typename F>
__device__
float3 lighting(float3 col, float3 rayDir, float3 pos, T vol,
		float tstepModifier, float deltaGradient, F texLookup, int timeStep)
{
  auto lights = getLightsDefault();
  
  /* const float3 lightPos3 = make_float3(2.f, -2.f, -2.f); */
  /* const float3 lightPos4 = make_float3(-2.f, 2.f, 2.f); */
  /* const float3 lightPos5 = make_float3(-2.f, 2.f, -2.f); */
  /* const float3 lightPos6 = make_float3(-2.f, -2.f, 2.f); */
  /* const float3 lightPos7 = make_float3(-2.f, -2.f, -2.f); */
      
      
      
  float3 ambientColor = col;
  
  //float3 gradient = gradientWithSobel3D(pos, deltaGradient);
  ////float3 gradient = gradientWithSobel3D(pos, 4*deltaGradient, vol);
  const float3 gradient = gradientWithCentralDifferences<texNormalized>(pos, deltaGradient, vol, texLookup);
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
__device__
void lightingCompositing(float4& sum, float& w, float4& col, float3 rayDir, float3 pos, T vol, float tstepModifier, float deltaGradient, F texLookup, int timeStep=0)
{              
  //w = 1.f - __powf(1.f-col.w, tstepModifier);      
  ////w = adjustOpacityContribution(col.w, tstepModifier);
          
  //if(col.w > 0.0001f)
  {
    const float3 colRGB =
      lighting<texNormalized>(make_float3(col), rayDir, pos, vol,
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
