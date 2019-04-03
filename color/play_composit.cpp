#include <cstdlib>
#include <helper_math.h>
#include <iostream>
#include "../volData/cimg_helper.h"
#include "../volData/resampleKernel.h"


inline __host__ __device__ double4 operator*(double b, double4 a)
{
  return make_double4(b * a.x, b * a.y, b*a.z, b*a.w);
}

inline __host__ __device__ double4 operator*(double4 a, double b)
{
  return make_double4(b * a.x, b * a.y, b*a.z, b*a.w);
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
  
  return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
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

#include "over.h"
#include <iostream>
#include <cassert>

/*
inline __host__ __device__ double4 make_double4(double x, double y, double z, double w)
{
  double4 d;
  d.x = x;
  d.y = y;
  d.z = z;
  d.w = w;
  return d;
}
*/

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

template<typename S4, typename I3, typename I>
void writeResample(std::string fname, const std::vector<S4>& oin, I3 dim, I nChannels, double reduction)
{
  int3 newDim = dim;
  newDim.x/=reduction;
  newDim.y/=reduction;
  auto newO = resampleKernel(oin, dim, newDim);
  write(fname,
        newO,
        newDim, nChannels);
}

int main(int argc, char** argv)
{
  /*
template<typename I3>
void cimgWrite(std::string fname, std::vector<double> data, const I3& dim, 
	       int nChannels, unsigned char flip=0)
   */

  /*
    template<typename T, typename I3>
void cimgRead(std::vector<T>& data, I3& volDim, int& nChannels, const std::string& fname, bool forceGray)
   */

  typedef double S;
  typedef double4 S4;

  //typedef float S;
  //typedef float4 S4;

  if(argc < 2)
    {
      std::cout << "Usage: [file name out] [in0] ... [inn]\n";
      return 0;
    }

  const std::string foutBase(argv[1]);
  std::vector<std::vector<S4>> iin4;
  int3 dim;
  int nChannels = 0;
    
  {
    
    std::vector<std::string> fins;
    for(size_t a=2; a<argc; a++)
      fins.push_back(argv[a]);

    iin4.resize(fins.size());
  
    

    for(size_t i=0; i<fins.size(); i++)
      {
        auto prev_dim = dim;
        auto prev_nChannels = nChannels;
        std::vector<S> iin;
        cimgRead(iin, dim, nChannels, fins[i]);

        assert(i==0 || nChannels == prev_nChannels);
        assert(i==0 || (dim.x == prev_dim.x && dim.y == prev_dim.y && dim.z == prev_dim.z));

        assert(nChannels == 4);

        const size_t nElems = dim.x*dim.y*dim.z;

        iin4[i].resize(nElems);
        for(size_t j=0; j<nElems; j++)
          {
            iin4[i][j].x = iin[j*4+0]/255.;
            iin4[i][j].y = iin[j*4+1]/255.;
            iin4[i][j].z = iin[j*4+2]/255.;
            iin4[i][j].w = iin[j*4+3]/255.;            
          }
      }        
  }

  {
    int3 newDim = dim;
    newDim.x *= 2;
    newDim.y *= 2;

    std::vector<S4> o(newDim.x*newDim.y);

    auto copyTile = [&](int i, int3 off)
      {
        int3 pos;
        pos.z = 0;
        for(pos.y=0; pos.y<dim.y; pos.y++)
          for(pos.x=0; pos.x<dim.x; pos.x++)
            { 
              o[iii2i(off+pos, newDim)] = iin4[i][iii2i(pos, dim)];
            }
      };

    copyTile(0, make_int3(0,0,0));
    copyTile(1, make_int3(dim.x,0,0));
    copyTile(2, make_int3(0,dim.y,0));
    copyTile(3, make_int3(dim.x,dim.y,0));

    write(foutBase+"_single.png",
          o, newDim, nChannels);

    writeResample(foutBase+"_single_small.png",o,newDim,nChannels, 8.);
  }

  {
    std::vector<S4> o(iin4.front().size());

    for(size_t j=0; j<o.size(); j++)
      o[j].x = o[j].y = o[j].z = o[j].w = 0.;
      
    for(auto img : iin4)
      {
        for(size_t j=0; j<o.size(); j++)
          {
            over(o[j], img[j]);
          }
      }

    
    
    write(foutBase+"_over.png",
          o, dim, nChannels);

    writeResample(foutBase+"_over_small.png", o, dim, nChannels, 4.);
    
    
  }

  {
    const double tstepModMod = 0.001;
    std::vector<S4> o(iin4.front().size());

    for(size_t j=0; j<o.size(); j++)
      o[j].x = o[j].y = o[j].z = o[j].w = 0.;
      
    for(auto img : iin4)
      {
        for(size_t j=0; j<o.size(); j++)
          {
            over(o[j], img[j], adjustOpacityContribution(img[j].w, tstepModMod));            
          }
      }

    for(size_t j=0; j<o.size(); j++)
      {
        if(o[j].w > 0.f)
	{
	  o[j].x /= o[j].w;
	  o[j].y /= o[j].w;
	  o[j].z /= o[j].w;
	  o[j].w = adjustOpacityContribution(o[j].w, 1./tstepModMod);
          }
      }
    write(foutBase+"_comp.png",
          o, dim, nChannels);

    writeResample(foutBase+"_comp_small.png",o,dim,nChannels, 4.);    
  }  
  
  return 0;
}
