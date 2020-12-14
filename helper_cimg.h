#ifndef __HELPER_CIMG__
#define __HELPER_CIMG__

#include <cstdlib>
#define cimg_display 0
#define cimg_use_png

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wnull-pointer-arithmetic"
#include <CImg.h>
#pragma GCC diagnostic pop

#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <volData/vec.h>
#include <ostream>
#include "helper/helper_idx.h"
#include "helper/helper_assert.h"

namespace helper
{

  template<typename I3>
  auto cimgIdx(
	       const size_t x,
	       const size_t y,
	       const size_t z,
	       const size_t c,
	       const I3& dim, const size_t /*nChannels*/)
  {
    return c*dim.x*dim.y*dim.z+z*dim.x*dim.y+y*dim.x+x;
  }
  
  template<typename I3>
  auto cimgConvert(unsigned char* data, const I3& dim, 
		   int nChannels, unsigned char flip=0)
  {
    const int defaultValue = 0;
        
    cimg_library::CImg<unsigned char> out(dim.x,dim.y,dim.z,nChannels,defaultValue);
    for(size_t z=0; z<dim.z; z++)
      for(size_t y=0; y<dim.y; y++)
	for(size_t x=0; x<dim.x; x++)
	  for(int c=0; c<nChannels; c++)
	    {
	      size_t vx = x;
	      if(flip & 1)
		vx = (dim.x-x-1);
	      size_t vy = y;
	      if(flip & 2)
		vy = (dim.y-y-1);
	      size_t vz = z;
	      if(flip & 4)
		vz = (dim.z-z-1);
	    
	      out.data()[cimgIdx(vx,vy,vz,c,dim, nChannels)] = 
		((unsigned char*) &data[0])[c+nChannels*(x+dim.x*(y+dim.y*z))]
	    
		//(c==0 || c==3)*255
		;
	    }

    return out;
  }


    template<typename I3>
    void cimgResize(unsigned char* dataOut, const I3& outDim,
		    unsigned char* data, const I3& dim, 
		    int nChannels, unsigned char flip=0)
    {
      auto out = cimgConvert(data, dim, nChannels, flip);
      out.resize(outDim.x, outDim.y, outDim.z, nChannels, 5);

      for(size_t z=0; z<outDim.z; z++)
	for(size_t y=0; y<outDim.y; y++)
	  for(size_t x=0; x<outDim.x; x++)
	    for(int c=0; c<nChannels; c++)
	      dataOut[c+nChannels*(x+outDim.x*(y+outDim.y*z))] =
		out.data()[cimgIdx(x,y,z,c,outDim, nChannels)];
    }
  
  
template<typename I3>
void cimgWrite(std::string fname, unsigned char* data, const I3& dim, 
	       int nChannels, unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  //uint2 imgDim = make_uint2(dim.x, dim.y);

  auto out = cimgConvert(data, dim, nChannels, flip);
  if((0 != outDim.x && 0 != outDim.y) && (dim.x != outDim.x || dim.y != outDim.y))
    {
      out.resize(outDim.x, outDim.y, outDim.z, nChannels, 5);
    }
                    
  out.save(fname.c_str()/*"bla.tiff"*/);
}


template<typename T, typename I3>
void cimgWrite(std::string fname, T* data, const I3& dim, 
	       int nChannels, unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  std::vector<unsigned char> d(dim.x*dim.y*dim.z*nChannels);
  for(size_t i=0; i<dim.x*dim.y*dim.z*nChannels; i++)
    {
      const unsigned char v =
	std::max(0., std::min(data[i]/**255.*/+.5, 255.));
      d[i] = v;
    }
  cimgWrite(fname, &d[0], dim, nChannels, flip, outDim);
}

  template<typename TV, typename I3>
void cimgWriteNorm(std::string fname, TV data_in, const I3& dim, 
	       int nChannels, unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{

  typedef typename std::remove_reference<decltype(data_in[0])>::type T;
    static_assert(std::is_floating_point<T>::value,
		"needs to be a floating point number");

    
  const auto n = helper::iii2n(dim)*nChannels;

  //std::vector<T> data(&data[0], (&data[0])+n);

  std::vector<T> data(n);

  for(size_t i=0; i<n; i++)
    data[i] = data_in[i]*255.;
  
  //for(auto it = &data[0]; it != (&data[0])+n; it++)
  //for(auto& e : data)
  //e *= 255.;
  
  cimgWrite(fname, &data[0], dim, nChannels, flip, outDim);
}

    template<typename T, typename I3>
    T cimgWriteNormalizeScalar(std::string fname, std::vector<T> data, const I3& dim, unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  static_assert(std::is_floating_point<T>::value, "needs to be a floating point number");

  const double maxV = *std::max_element(data.begin(), data.end());
  
  for(auto& e : data)
    e = (e/maxV)*255.;
  
  cimgWrite(fname, &data[0], dim, 1, flip, outDim);
  return maxV;
}

  template<typename VT, typename I>
void cimgWriteNormRGBA(std::string fname, std::vector<VT> dataVec, const V3<I>& dim, 
		       unsigned char flip=0, const V3<I>& outDim=make_vec<V3<I>>(0,0,0))
{
  const size_t nChannels = 4;

  assert(!dataVec.empty());
  typedef decltype(dataVec[0].x) T;
  std::vector<T> data(dataVec.size()*nChannels);

  for(uint64_t i=0; i<dataVec.size(); i++)
    {
      data[i*nChannels+0] = dataVec[i].x;
      data[i*nChannels+1] = dataVec[i].y;
      data[i*nChannels+2] = dataVec[i].z;
      data[i*nChannels+3] = dataVec[i].w;
    }
  
  cimgWriteNorm(fname, data, dim, nChannels, flip, outDim);
}

  template<typename VT, typename I>
  void cimgWriteNormRGBA(std::string fname, std::vector<VT> dataVec, const V2<I>& dim)
  {
    cimgWriteNormRGBA(fname, dataVec, V3<I>(dim.x, dim.y, 1)); 
  }

      template<typename VT, typename I3>
void cimgWriteNormRGB(std::string fname, std::vector<VT> dataVec, const I3& dim, 
		       unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  const size_t nChannels = 3;

  assert(!dataVec.empty());
  typedef decltype(dataVec[0].x) T;
  std::vector<T> data(dataVec.size()*nChannels);

  for(uint64_t i=0; i<dataVec.size(); i++)
    {
      data[i*nChannels+0] = dataVec[i].x;
      data[i*nChannels+1] = dataVec[i].y;
      data[i*nChannels+2] = dataVec[i].z;
    }
  
  cimgWriteNorm(fname, data, dim, nChannels, flip, outDim);
}

 

  
template<typename T, typename I3>
void cimgWrite(std::string fname, std::vector<T> data, const I3& dim, 
	       int nChannels, unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  cimgWrite(fname, &data[0], dim, nChannels, flip, outDim);
}

       template<typename VT, typename I3>
void cimgWriteRGB(std::string fname, std::vector<VT> dataVec, const I3& dim, 
		  unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  const size_t nChannels = 3;

  assert(!dataVec.empty());
  typedef decltype(dataVec[0].x) T;
  std::vector<T> data(dataVec.size()*nChannels);

  for(uint64_t i=0; i<dataVec.size(); i++)
    {
      data[i*nChannels+0] = dataVec[i].x;
      data[i*nChannels+1] = dataVec[i].y;
      data[i*nChannels+2] = dataVec[i].z;
    }
  
  cimgWrite(fname, data, dim, nChannels, flip, outDim);
}

  template<typename VT, typename I3>
  void cimgWriteRGBA(std::string fname, const VT& dataVec, const I3& dim, 
		     unsigned char flip=0, const I3& outDim=make_vec<I3>(0,0,0))
{
  const size_t nChannels = 4;

  typedef decltype(dataVec[0].x) T;
  const auto nElems = helper::iii2n(dim);
  std::vector<T> data(nElems*nChannels);

  for(uint64_t i=0; i<nElems; i++)
    {
      data[i*nChannels+0] = dataVec[i].x;
      data[i*nChannels+1] = dataVec[i].y;
      data[i*nChannels+2] = dataVec[i].z;
      data[i*nChannels+3] = dataVec[i].w;
    }
  
  cimgWrite(fname, data, dim, nChannels, flip, outDim);
}
  
template<typename T, typename I3>
  void cimgRead(std::vector<T>& data, I3& volDim, int& nChannels, const std::string& fname, bool forceGray=false, bool forceDim=false)
{  
  cimg_library::CImg<T> img(fname.c_str());

  if(forceDim && (volDim.x != img.width() || volDim.y != img.height()))
    img.resize(volDim.x, volDim.y, volDim.z, img.spectrum(), 5);
  
  volDim.x = img.width();
  volDim.y = img.height();
  volDim.z = img.depth();  
  nChannels = img.spectrum();

  assert(volDim.z == 1);
  //std::cout << fname << ", there are #channels: " << nChannels << std::endl;
  if(forceGray)
    nChannels = 1;
  data.resize(volDim.x*volDim.y*volDim.z*nChannels);

  size_t idx =0;
  for(size_t z=0; z<(size_t)volDim.z; z++)
    for(size_t y=0; y<(size_t)volDim.y; y++)
      for(size_t x=0; x<(size_t)volDim.x; x++)
	for(size_t c=0; c<(size_t)nChannels; c++)
	  {
	    data[idx] = img(x,y,z,c);
	    bool allChannelsEqual = true;
	    if(img.spectrum() > 1)
	      allChannelsEqual = allChannelsEqual && (img(x,y,z,0)==img(x,y,z,1));
	    if(img.spectrum() > 2)
	      allChannelsEqual = allChannelsEqual && (img(x,y,z,0)==img(x,y,z,2));
	    if(img.spectrum() > 3)
	      allChannelsEqual = allChannelsEqual && (img(x,y,z,0)==img(x,y,z,3));
	    
	    /*
	    if(!allChannelsEqual)
	      std::cout << (int)img(x,y,z,0) << " " << (int)img(x,y,z,1) << " " << (int)img(x,y,z,2) << std::endl;
	    */

	    hassert(!forceGray || allChannelsEqual);
	    idx++;
	  }
}
};

#endif //__HELPER_CIMG__
