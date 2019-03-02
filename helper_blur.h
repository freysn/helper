#ifndef __HELPER_BLUR__
#define __HELPER_BLUR__

#include "helper_util.h"

namespace helper
{

  const size_t blurKernel_size = 1024;
#ifdef USE_CUDA_RUN_DEVICE
  __constant__
  #endif
  double blurKernel[blurKernel_size];
  
  // http://blog.ivank.net/fastest-gaussian-blur.html
  
  // source channel, target channel, width, height, radius
  template<typename T, typename TV0, typename TV1>
#ifdef __CUDACC__
  __host__ __device__
#endif
  void gaussBlur_1(const TV0& scl, TV1& tcl, int64_t w, int64_t h, /*T r,*/ int64_t i, int64_t j, /*const T* kernel,*/ int64_t rs)
  {

#ifndef __CUDACC__
    using namespace std;
#endif
  
    typedef int64_t I;
    //I rs = ceil(r * 2.57);     // significant radius
    
    //T val = static_cast<I>(0), wsum = static_cast<I>(0);
    T val = 0.;
    I idx =0 ;
    for(I iy = i-rs; iy<i+rs+1; iy++)
      for(I ix = j-rs; ix<j+rs+1; ix++) {
	const I x = helper::min(w-1, helper::max(static_cast<I>(0), ix));
	const I y = helper::min(h-1, helper::max(static_cast<I>(0), iy));
	//I dsq = (ix-j)*(ix-j)+(iy-i)*(iy-i);
	//T wght = exp( -dsq / (2*r*r) ) / (M_PI*2*r*r);
	T wght = blurKernel[idx];
	val += scl[y*w+x] * wght;
	//wsum += wght;
	idx++;
      }
    tcl[i*w+j] = val;// /wsum;    
  }

  /*
  template<typename TV, typename T>
#ifdef __CUDACC__
  __host__ __device__
#endif
  void gaussBlur_1(const TV& scl, TV& tcl, int64_t w, int64_t h, T r)
  {
    typedef int64_t I;
    for(auto i=static_cast<I>(0); i<h; i++)
      for(auto  j=static_cast<I>(0); j<w; j++)
	gaussBlur_1(scl, tcl, w, h, r, i, j);
  }
  */
  
  template<typename F>
  struct BlurFunctor
  {

    void setInOut(F* in_in, F* out_in)
    {
      in = in_in;
      out = out_in;
    }

    //const F* in_in, F* out_in,
    // :
    //in(in_in), out(out_in),
    
    bool updateBlurKernel(const F r_in)
    {
      if(r==r_in)
	return false;
      r = r_in;
      static_assert(std::is_same<F, std::remove_reference<decltype(blurKernel[0])>::type>::value, "use same type for blur kernel as for input fields");
      
      typedef int64_t I;
      I rs = ceil(r * 2.57);     // significant radius

      std::vector<F> kernel;
      
      F wsum = 0.;
      for(I iy = -rs; iy<rs+1; iy++)
	for(I ix = -rs; ix<rs+1; ix++)
	  {
	    I dsq = (ix)*(ix)+(iy)*(iy);
	    F wght = exp( -dsq / (2*r*r) ) / (M_PI*2*r*r);
	    kernel.push_back(wght);
	    wsum += wght;
	  }
      for(auto & e: kernel)
	e /= wsum;

      if(blurKernel_size < kernel.size())
	{
	  std::cerr << "ERROR: THE BLUR KERNEL IS TOO SMALL TO CAPTURE THE SPECIFIED RADIUS " << r << std::endl;
	  exit(-1);
	}

      #ifdef USE_CUDA_RUN_DEVICE
      //static_assert(false);
      cudaMemcpyToSymbol(blurKernel, &kernel[0], sizeof(F)*kernel.size());
      assert(cudaGetLastError() == cudaSuccess);
      #else
      memcpy(&blurKernel[0], &kernel[0], sizeof(F)*kernel.size());
      #endif

      std::cout << "the kernel has " << kernel.size() << "entries\n";
      return true;
    }
    
#ifdef __CUDACC__
    __host__ __device__
#endif
    void operator()(uint32_t idx) const
    {
      auto j=idx/w;
      auto i = idx - w*j;
      helper::gaussBlur_1<F>(in, out, w, h, /*r,*/ i, j, /*&kernel[0],*/ ceil(r * 2.57));
    }
		  
    int64_t w;
    int64_t h;
    F r=-1.;
    F* in;
    F* out;
    //std::vector<F> kernel;
  };
}

#endif //__HELPER_BLUR__
