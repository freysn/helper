#ifndef __HELPER_UTIL__
#define __HELPER_UTIL__

#include "color/over.h"
#include "volData/vec.h"
#include <unistd.h>
#include <limits>
#include <vector>
#include <random>

namespace helper
{
  template<typename T0, typename T1>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T0 mix(const T0& x, const T0& y,  const T1& a)
  {
    return x*(1-a)+y*a;
  }

  template<typename T>
#ifdef __CUDACC__
  __device__ __host__
#endif
  const T max(const T a, const T b)
  {
    return (a < b) ? b : a;
  }

  template<typename T>
#ifdef __CUDACC__
  __device__ __host__
#endif
  const T max(const T a, const T b, const T c)
  {
    return max(a, max(b,c));
  }
  

  template<typename T>
#ifdef __CUDACC__
  __device__ __host__
#endif
  const T min(const T a, const T b)
  {
    return (b < a) ? b : a;
  }
  
  V4<double> premultiply(V4<double> a)
  {
    a.x *= a.w;
    a.y *= a.w;
    a.z *= a.w;

    return a;
  }

  std::vector<V4<double>> composit(const std::vector<V4<double>>& a,  const std::vector<V4<double>>& b, bool premultiplied=true)
  {
    assert(a.size() == b.size());
    std::vector<V4<double>> out(a.size());

    for(size_t i=0; i<out.size(); i++)
      {
	out[i] = a[i];

	if(!premultiplied)
	  out[i] = premultiply(a[i]);
	
	over(out[i], b[i]);
      }
    return out;
  }

  size_t getTotalSystemMemory()
  {
    size_t pages = sysconf(_SC_PHYS_PAGES);
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
  }

  template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  void swap(T& a, T& b)
  {
    T c(a);
    a=b;
    b=c;
  }

  template<typename T0, typename T1, typename T2>
#ifdef __CUDACC__
  __device__ __host__
#endif
  T0 clamp(T0 v, T1 vmin, T2 vmax)
  {
#ifndef __CUDACC__
    using namespace std;
#endif
    return max(T0(vmin), min(v, T0(vmax)));    
  }

  template<typename T, typename V>
#ifdef USE_CUDA_RUN_DEVICE
  __host__ __device__
#endif
  void rgbaFloatToInt(T& x, T& y, T &z, T& w, V rgba)
    {
      
    auto clamp01 = [](auto f)
      {
#ifdef USE_CUDA_RUN_DEVICE
	return __saturatef(f);
#else
	return std::max(static_cast<decltype(f)>(0), std::min(f, static_cast<decltype(f)>(1)));
#endif
      };
    

    // rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    // rgba.y = __saturatef(rgba.y);
    // rgba.z = __saturatef(rgba.z);
    // rgba.w = __saturatef(rgba.w);

    x = uint(clamp01(rgba.x)*255);
    y = uint(clamp01(rgba.y)*255);
    z = uint(clamp01(rgba.z)*255);
    w = uint(clamp01(rgba.w)*255);
    }

  template<typename V>
#ifdef USE_CUDA_RUN_DEVICE
  __host__ __device__
#endif
  uint rgbaFloatToInt(V rgba)
  {

    uint x,y,z,w;
    rgbaFloatToInt(x,y,z,w,rgba);

    return (w<<24) | (z<<16) | (y<<8) | x;
  }

  template<typename F>
#ifdef __CUDACC__
  __device__ __host__
#endif
  F eps()
  {
    //return std::numeric_limits<F>::epsilon();
    return 1.e-15;
  }

  template<typename F>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool apprEq(const F a, const F b, const F epsf = helper::eps<F>())
  {
    return (a+epsf >= b) && (b+epsf  >= a);
  }


    template<typename F>
    void assert_norm(const V4<F>& e, const F epsf = helper::eps<F>())
  {
    assert(e.x >= -epsf && e.x <= 1.+epsf);
    assert(e.y >= -epsf && e.z <= 1.+epsf);
    assert(e.z >= -epsf && e.z <= 1.+epsf);
    assert(e.w >= -epsf && e.w <= 1.+epsf);
  };

  template<typename F, typename T, typename DIST>
  auto selectDifferentEntries(const std::vector<T>& values, const DIST& dist, size_t nSelect)
  {
    

    const int64_t selectAfterNIterations = 1;

    std::vector<size_t> out;
    out.reserve(nSelect);

    {      
      std::mt19937 rng(1337);
      std::uniform_int_distribution<size_t> uniform_dist(0, values.size()-1);
      out.push_back(uniform_dist(rng));
    }
    
    for(int64_t i=-selectAfterNIterations; i<static_cast<int64_t>(nSelect)-1; i++)
      {
	//std::cout << "i: " << i << std::endl;
	if(i==0)
	  {
	    assert(!out.empty());
	    const auto init = out.back();
	    out.clear();
	    out.push_back(init);
	    //std::cout << "clear\n";
	  }

	//
	// find value with largest minimum distance to current set
	//
	std::pair<F, size_t> maxMinDist(0., 0);
	
	for(size_t j=0; j<values.size(); j++)
	  {
	    F minDist = std::numeric_limits<F>::max();
	    for(const auto e : out)
	      minDist = helper::min(static_cast<F>(dist(values[j], values[e])), minDist);

	    if(minDist > maxMinDist.first)
	      {
		maxMinDist.first = minDist;
		maxMinDist.second = j;
	      }
	  }

	out.push_back(maxMinDist.second);
	//std::cout << "out.size(): " << out.size() << std::endl;
      }
    
    return out;
  }

  template<typename F>
#ifdef USE_CUDA_RUN_DEVICE
  __host__ __device__
#endif
  F sqrt(F v)
  {
#ifdef  __CUDA_ARCH__
    return sqrtf(v);
#else
    return std::sqrt(v);
#endif
  }

  template<typename T=std::vector<size_t>>
  T range_bn(int64_t begin, int64_t n)
  {
    assert(n>=0);
    if(n==0)
      return T();
    
    T a(n);
    std::iota(a.begin(), a.end(), begin);
    return a;
  }

  template<typename T=std::vector<size_t>>
  T range_be(int64_t begin, int64_t end)
  {
    assert(begin <= end);
    return range_bn<T>(begin,end-begin);
  }

  template<typename T=std::vector<size_t>>
  T range_n(int64_t n)
  {
    return range_bn<T>(0,n);
  }
  
  template<typename T=std::vector<size_t>>
  T rangeVec(int64_t n, int64_t start=0)
  {
    T a(n);
    std::iota(a.begin(), a.end(), start);
    return a;
  }  

  template<typename T>
  T range(int64_t n, int64_t start=0)
  {
    const auto tmp = rangeVec<std::vector<typename T::value_type>>(n, start);
    return T(tmp.begin(), tmp.end());
  }
  

  template<typename T_it>
  auto normalize(T_it b, T_it e)
  {
    const auto mm = std::minmax_element(b, e);

    const auto minv = *mm.first;
    const auto maxv = *mm.second;
    
    assert(minv < maxv);
    for(auto it=b; it != e; it++)
      *it = (*it-minv)/(maxv-minv);
    
    return std::make_pair(minv, maxv);
  }
  
  template<typename T>
  T iDivUp(T a, T b) 
  { 
    static_assert(std::is_integral<T>::value, "Integral required.");
    return (a % b != 0) ? (a / b + 1) : (a / b); 
  }
};


#endif // __HELPER_UTIL__
