#ifndef __HELPER_BLUR__
#define __HELPER_BLUR__

#include "helper_util.h"

namespace helper
{
  
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
  
  
  struct BlurFunctor
  {
    BlurFunctor(){}

    BlurFunctor(int64_t w_in, int64_t h_in, double r_in) :
      w(w_in), h(h_in)
    {
      updateBlurKernel(r_in);
    }
    
    // void setInOut(F* in_in, F* out_in)
    // {
    //   in = in_in;
    //   out = out_in;
    // }

    //const F* in_in, F* out_in,
    // :
    //in(in_in), out(out_in),

    size_t updateBlurKernel(const double r_in)
    {
      if(r==r_in)
	return false;
      r = r_in;
      //static_assert(std::is_same<F, std::remove_reference<decltype(blurKernel[0])>::type>::value, "use same type for blur kernel as for input fields");
      
      typedef int64_t I;
      I rs = ceil(r * 2.57);     // significant radius

      std::vector<double> kernel;

      const auto rsy = (h==1) ? 0 : rs;
      
      double wsum = 0.;
      for(I iy = -rsy; iy<rsy+1; iy++)
	for(I ix = -rs; ix<rs+1; ix++)
	  {
	    I dsq = (ix)*(ix)+(iy)*(iy);
	    double wght = exp( -dsq / (2*r*r) ) / (M_PI*2*r*r);
	    kernel.push_back(wght);
	    wsum += wght;
	  }
      for(auto & e: kernel)
	e /= wsum;

      
      #ifdef USE_CUDA_RUN_DEVICE
      {
	if(blurKernel_size < kernel.size())
	  {
	    std::cerr << "ERROR: THE BLUR KERNEL IS TOO SMALL TO CAPTURE THE SPECIFIED RADIUS " << r << std::endl;
	    exit(-1);
	    return 0;
	  }

	const auto nBytes = sizeof(double)*kernel.size();
	//static_assert(false);
	cudaMemcpyToSymbol(blurKernel, &kernel[0], nBytes);
	assert(cudaGetLastError() == cudaSuccess);
      }
      #else
      blurKernel=kernel;
      //memcpy(&blurKernel[0], &kernel[0], nBytes);
      #endif

      return kernel.size();
    }

    template<typename TV0>
#ifdef __CUDACC__
    __host__ __device__
#endif
    auto operator()(uint32_t idx, const TV0& in) const
    {
      auto j=idx/w;
      auto i = idx - w*j;

      assert(i<w);
      assert(j<h);
      
      return gaussBlur_1(in, w, h, /*r,*/ i, j, /*&kernel[0],*/ ceil(r * 2.57));
    }

    // http://blog.ivank.net/fastest-gaussian-blur.html
  
    // source channel, target channel, width, height, radius
    template</*typename T,*/ typename TV0>
#ifdef __CUDACC__
    __host__ __device__
#endif
    auto gaussBlur_1(const TV0& scl, int64_t w, int64_t h, /*T r,*/ int64_t i, int64_t j, /*const T* kernel,*/ int64_t rs) const
    {

#ifndef __CUDACC__
      using namespace std;
#endif
  
      typedef int64_t I;
      //I rs = ceil(r * 2.57);     // significant radius
    
      //T val = static_cast<I>(0), wsum = static_cast<I>(0);
      auto val = scl[0]*0;
      I idx =0 ;

      const auto rsy = (h==1) ? 0 : rs;
      
      for(I iy = j-rsy; iy<j+rsy+1; iy++)
	for(I ix = i-rs; ix<i+rs+1; ix++) {
	  const I x = helper::min(w-1, helper::max(static_cast<I>(0), ix));
	  const I y = helper::min(h-1, helper::max(static_cast<I>(0), iy));
	  //I dsq = (ix-j)*(ix-j)+(iy-i)*(iy-i);
	  //T wght = exp( -dsq / (2*r*r) ) / (M_PI*2*r*r);
	  double wght = blurKernel[idx];
	  val += scl[y*w+x] * wght;
	  //wsum += wght;
	  idx++;

	  //std::cout << "blurKernel[" << idx << "]=" << wght << " scl[" << x << "," << y << "]=" << scl[y*w+x].x << " update: " << val.x << " w: " << w << " h: " << h << " i: " << i << " j: " << j << " ix: " << ix << " iy: " << iy << std::endl;
	}
      //tcl[j*w+i] = val;// /wsum;
      return val;
    }

		  
    const int64_t w=0;
    const int64_t h=0;
    double r=-1.;
    //F* in;
    //F* out;
    //std::vector<F> kernel;

#ifdef USE_CUDA_RUN_DEVICE
    
    static const size_t blurKernel_size =
      //1024*1024
      256*256
      ;
    __constant__ double blurKernel[blurKernel_size];
#else
    std::vector<double> blurKernel;
#endif
    

  };
}

#endif //__HELPER_BLUR__
