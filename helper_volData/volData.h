#ifndef __VOL_DATA__
#define __VOL_DATA__

#include <iostream>
#include <cmath>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "helper/helper_writeFile.h"
#include "helper/helper_volData/vec.h"

using namespace std;

enum elemType_t {UCHAR, USHORT, FLOAT};

typedef unsigned int mm_uint;
typedef unsigned short mm_ushort;
typedef unsigned char mm_uchar;

//mm_ushort* loadRawFile(char *filename, size_t size, elemType_t elemType);



// template<typename T>
// class vec3
// {
// public:
//   vec3()
//   {
//   }
  
//   vec3(T x, T y, T z)
//   {
//     this->x = x;
//     this->y = y;
//     this->z = z;
//   }

//   T& operator[](unsigned idx)
//   {
//     assert(/*idx >= 0 &&*/ idx <= 2);

//     if(idx == 0)
//       return x;
//     else if(idx == 1)
//       return y;					
//     else
//       return z;
//   }

//   T x;
//   T y;
//   T z;
// };

typedef V3<mm_uint> mm_uint3;
typedef V3<int> mm_int3;
typedef V3<float> mm_float3;

/*
  -resize data to fit certain size
  -convert data to other data types
 */

mm_uint3 resizeDim(mm_uint3 from, size_t nMaxElements);

#if __cplusplus<201703L
template<typename T>
T clamp(T v, T n, T x)
{
  return std::max(std::min(v, x), n);
}
#endif

template<typename F, typename T, typename I3>
double trilinear(V3<F> c, T* data, I3 dataDim)
{
  c.x = std::max((F)0., std::min(c.x, (F)(dataDim.x-1)));
  c.y = std::max((F)0., std::min(c.y, (F)(dataDim.y-1)));
  c.z = std::max((F)0., std::min(c.z, (F)(dataDim.z-1)));
  
  mm_uint3 minNeighbors;
  
  for(int i=0; i<3; i++)
    {
      minNeighbors[i] = floor(c[i]);
      c[i] -= minNeighbors[i];
    }
  
  mm_uint3 iter;
  double value = 0.;
  for(iter.z=0; iter.z<2; iter.z++)
    for(iter.y=0; iter.y<2; iter.y++)
      for(iter.x=0; iter.x<2; iter.x++)
	{
	  float weight = 1.f;
	  for(int i=0; i<3; i++)
	    if(iter[i] == 0)
	      weight *= (1.f-c[i]);
	    else
	      weight *= c[i];
	  int x = (minNeighbors.x+iter.x);
	  int y = (minNeighbors.y+iter.y);
	  int z = (minNeighbors.z+iter.z);


	  x =
#if __cplusplus>=201703L
	    std::
#endif
	    clamp(x, (int)0, (int)dataDim.x-1);
	  y =
#if __cplusplus>=201703L
	    std::
#endif
	    clamp(y, (int)0, (int)dataDim.y-1);
	  z =
#if __cplusplus>=201703L
	    std::
#endif
	    clamp(z, (int)0, (int)dataDim.z-1);
	  
	  value += weight * data[x + dataDim.x*(y+dataDim.y*z)]; 
	}
  
  return value;
  //return data[minNeighbors.x+dataDim.x*(minNeighbors.y+dataDim.y*minNeighbors.z)];
}

template<typename T>
void resample(T* to, mm_uint3 toDim, T* fromData, mm_uint3 fromDim)
{
  mm_float3 scale;
  for(int i=0; i<3; i++)
    scale[i] = (float)fromDim[i]/(float)toDim[i];
  
  std::cout << "scale: " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
  
  size_t cnt = 0;
  for(decltype(toDim.z) z=0; z<toDim.z; z++)
    {
      //std::cout << z << " of " << toDim.z << std::endl;
      for(decltype(toDim.y) y=0; y<toDim.y; y++)
	for(decltype(toDim.x) x=0; x<toDim.x; x++)
	  {
	    to[cnt] = trilinear(mm_float3(scale.x*x, scale.y*y, scale.z*z), fromData, fromDim);
	    cnt++;
	  }
    }
}



mm_ushort *loadRawFileUShort(char *filename, size_t size, size_t offset=0);
mm_ushort *loadRawFileUChar(char *filename, size_t size, size_t offset=0);
unsigned char* loadRawFileUCharPlain(char *filename, size_t size, size_t offset=0);
mm_ushort *loadRawFileFloat(char *filename, size_t size, size_t offset=0);
mm_ushort* loadRawFile(char *filename, size_t size, elemType_t elemType, size_t offset=0);

#endif /*__VOL_DATA__*/
