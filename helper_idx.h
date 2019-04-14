#ifndef __VOL_IDX_HELPER2__
#define __VOL_IDX_HELPER2__

#include <list>
#include <cstddef>
#include <cassert>


namespace helper
{

  template<typename I2A, typename I2B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t ii2i(const I2A& v, const I2B& volDim)
{
  typedef size_t I;
  return (I)v.x+volDim.x*(I)v.y;
}
  
template<typename I3A, typename I3B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iii2i(const I3A& v, const I3B& volDim)
{
  typedef size_t I;
  assert(v.x < volDim.x && v.y < volDim.y && v.z < volDim.z);
  return (I)v.x+volDim.x*((I)v.y+volDim.y*(I)v.z);
}

    template<typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  void iiizeros(I3& v)
    {
      v.x = 0;
      v.y = 0;
      v.z = 0;
    }
      
  template<typename I3A, typename I3B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  bool iiinext(I3A& v, const I3B& volDim)
{
  assert(v.x < volDim.x && v.y < volDim.y && v.z < volDim.z);
  v.x++;
  if(v.x == volDim.x)
    {
      v.x=0;
      v.y++;
      if(v.y == volDim.y)
	{
	  v.y=0;
	  v.z++;
	  if(v.z == volDim.z)
	    return false;
	}
    }
  return true;
}


  template<typename I4A, typename I4B>
#ifdef __CUDACC__
  __host__ __device__
#endif  
  bool iiiinext(I4A& v, const I4B& volDim)
  {
    assert(v.x < volDim.x && v.y < volDim.y && v.z < volDim.z);
    v.x++;
    if(v.x == volDim.x)
      {
	v.x=0;
	v.y++;
	if(v.y == volDim.y)
	  {
	    v.y=0;
	    v.z++;
	    if(v.z == volDim.z)
	      {
		v.z=0;
		v.w++;
		if(v.w == volDim.w)
		  return false;
	      }
	  }
      }
    return true;
  }



  
#if 0
template<typename I3>
  size_t iii2n(const I3& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y*(I)volDim.z;
}

template<>
  size_t iii2i(const std::pair<unsigned int, std::list<unsigned int> >& v, const int3& volDim)
{
  return v.first;
}
#endif

template<typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iii2n(const I3& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y*(I)volDim.z;
}

  template<typename I4>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iiii2n(const I4& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y*(I)volDim.z*(I)volDim.w;
}

  template<typename I4A, typename I4B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iiii2i(const I4A& v, const I4B& volDim)
{
  typedef size_t I;
  return (I)v.x+volDim.x*((I)v.y+volDim.y*((I)v.z + (I)v.w*volDim.z));
}



  template<typename I2>
        #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t ii2n(const I2& volDim)
{
  typedef size_t I;
  return (I)volDim.x*(I)volDim.y;
}

  template<typename T>
        #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t v2n(const V2<T>& volDim)
  {
    return ii2n(volDim);
  }

  template<typename T>
        #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t v2n(const V3<T>& volDim)
  {
    return helper::iii2n(volDim);
  }



/*
  uint32_t iii2i(const uint32_t& v, const int3& volDim)
  {
    return v;
  }
*/

  template<typename I, typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  I3 i2iii(I v, const I3 volDim)
{
  I3 out;
  out.z = v/(volDim.x*volDim.y);
  v -= out.z*volDim.x*volDim.y;
  out.y = v/volDim.x;
  out.x = v-out.y*volDim.x;

  //std::cout << __func__ << " " << v << " " << volDim.x << " " << volDim.y << " " << volDim.z << std::endl;
  assert(out.x < volDim.x
	 && out.y < volDim.y
	 && out.z < volDim.z);
  return out;
}

  template<typename I, typename I4>
#ifdef __CUDACC__
  __host__ __device__
#endif  
  I4 i2iiii(I v, const I4 volDim)
  {
    I4 out;
    auto n = (volDim.x*volDim.y*volDim.z);
    out.w = v/n;
    v-= out.w*n;
    out.z = v/(volDim.x*volDim.y);
    v -= out.z*volDim.x*volDim.y;
    out.y = v/volDim.x;
    out.x = v-out.y*volDim.x;

    //std::cout << __func__ << " " << v << " " << volDim.x << " " << volDim.y << " " << volDim.z << std::endl;
    assert(out.x < volDim.x
	   && out.y < volDim.y
	   && out.z < volDim.z
	   && out.w < volDim.w);
    return out;
  }

   template<typename I, typename I3>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  I3 i2ii(I v, const I3 volDim)
{
  I3 out;
  out.y = v/volDim.x;
  out.x = v-out.y*volDim.x;

  //std::cout << __func__ << " " << v << " " << volDim.x << " " << volDim.y << " " << volDim.z << std::endl;
  assert(out.x < volDim.x
	 && out.y < volDim.y
	 );
  return out;
}

  
  
  template<typename T0, typename T1>
     #ifdef __CUDACC__
  __host__ __device__
#endif  
  V2<T1> i2v(const T0& v, const V2<T1>& volDim)
  {
    static_assert(std::is_integral<T0>::value, "Integer required.");
    return i2ii(v, volDim);
  }

  template<typename T0, typename T1>
     #ifdef __CUDACC__
  __host__ __device__
#endif  
  V3<T1> i2v(const T0& v, const V3<T1>& volDim)
  {
    static_assert(std::is_integral<T0>::value, "Integer required.");
    return helper::i2iii(v, volDim);
  }

  template<typename I2A, typename I2B>
#ifdef __CUDACC__
  __host__ __device__
#endif  
  I2A iiclamp(const I2A& v, const I2B& volDim)
  {
#ifndef __CUDACC__
    using namespace std;
#endif  
    I2A o;
    o.x = max((decltype(v.x))0, min(v.x, (decltype(v.x))(volDim.x-1)));
    o.y = max((decltype(v.y))0, min(v.y, (decltype(v.y))(volDim.y-1)));
    return o;
  }
  
template<typename I3A, typename I3B>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  I3A iiiclamp(const I3A& v, const I3B& volDim)
  {
#ifndef __CUDACC__
    using namespace std;
#endif  
    I3A o;
    o.x = max((decltype(v.x))0, min(v.x, (decltype(v.x))(volDim.x-1)));
    o.y = max((decltype(v.y))0, min(v.y, (decltype(v.y))(volDim.y-1)));
    o.z = max((decltype(v.z))0, min(v.z, (decltype(v.z))(volDim.z-1)));
    //o.y = max(0, min(v.y, volDim.y-1));
    //o.z = max(0, min(v.z, volDim.z-1));
    return o;
  }

  template<typename I3, typename I3B>
bool isWithinBounds(const I3& v, const I3B& volDim)
{
  return
    v.x >= 0 && v.x < volDim.x
    && v.y >= 0 && v.y < volDim.y
    && v.z >= 0 && v.z < volDim.z;
}

  template<typename I2>
      #ifdef __CUDACC__
  __host__ __device__
#endif  
  bool iiWithinBounds(const I2& v, const I2& volDim, const I2 begin=I2(0,0))
{
  return
    v.x >= begin.x && v.x < volDim.x && v.y >= begin.y && v.y < volDim.y;
}


  template<typename I2A, typename I2B>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t ii2i_clamp(const I2A& v, const I2B& volDim)
  {
    return helper::ii2i(helper::iiclamp(v, volDim), volDim);
  }

template<typename I3A, typename I3B>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t iii2i_clamp(const I3A& v, const I3B& volDim)
  {
    return helper::iii2i(helper::iiiclamp(v, volDim), volDim);
  }

  template<typename T, typename I3B>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t v2i_clamp(const V3<T>& v, const I3B& volDim)
  {
    return iii2i_clamp(v, volDim);;
  }

    template<typename T, typename I3B>
    #ifdef __CUDACC__
  __host__ __device__
#endif  
  size_t v2i_clamp(const V2<T>& v, const I3B& volDim)
  {
    return ii2i_clamp(v, volDim);;
  }

}
#endif //__VOL_IDX_HELPER__
