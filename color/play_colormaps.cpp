#include <cstdlib>
#include <helper_math.h>
#include <iostream>
#include <cassert>
#include "../volData/cimg_helper.h"

inline __host__ __device__ double4 operator*(double b, double4 a)
{
  return make_double4(b * a.x, b * a.y, b*a.z, b*a.w);
}

inline __host__ __device__ double3 operator*(double b, double3 a)
{
  return make_double3(b * a.x, b * a.y, b*a.z);
}

inline __host__ __device__ double4 operator*(double4 a, double b)
{
  return make_double4(b * a.x, b * a.y, b*a.z, b*a.w);
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
  
  return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
  
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ void operator/=(double4 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline __host__ __device__ double4 make_double4(double b)
{
  double4 a;
    a.x = b;
    a.y = b;
    a.z = b;
    a.w = b;
    return a;
}

inline __host__ __device__ double4 make_double4(double3 b0, double b1)
{
  double4 a;
    a.x = b0.x;
    a.y = b0.y;
    a.z = b0.z;
    a.w = b1;
    return a;
}

#include "colormaps.h"

template<typename S4, typename I3, typename I>
void write(std::string fname, const std::vector<S4>& oin, I3 dim, I nChannels)
{
  assert(!oin.empty());
  auto o = oin;
  typedef decltype(o[0].x) S;
  for(size_t j=0; j<o.size(); j++)
      {
        o[j].x *= 255.;
        o[j].y *= 255.;
        o[j].z *= 255.;
        o[j].w *= 255.;
      }
    
    cimgWrite(fname,
              (S*) &o[0], dim, nChannels);
}


int main(int, char**)
{
  int3 dim;
  dim.x = 1024;
  dim.y = 512;
  dim.z = 1;
  std::vector<double4> img(dim.x*dim.y, make_double4(0.));

  int tileHeight = 64;
  int nColsStart = 2;  

  for(int i=0; i < dim.y/tileHeight; i++)
    {      
      const int nCols  = nColsStart+i;
      int tileWidth = dim.x/nCols;
      for(int j=0; j<nCols; j++)
        {
          double3 col = colormapNorm0<double3>((double)j/(nCols-1));

          int3 p;
          p.z = 1;
          for(p.y=i*tileHeight; p.y<(i+1)*tileHeight; p.y++)
            for(p.x=j*tileWidth; p.x<(j+1)*tileWidth; p.x++)
              {
                //std::cout << "p " << p.x << " " << p.y << std::endl;
                assert(p.x < dim.x);
                assert(p.y < dim.y);
                img[p.x+dim.x*p.y] = make_double4(col, 1.);                
              }
          
        }
    }

  write("colormap.png", img, dim, 4);
  
  return 0;
}
