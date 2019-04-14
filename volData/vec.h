#ifndef __VEC__
#define __VEC__

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <cassert>

/*
struct int3
  {
    int x;
    int y;
    int z;
  };

struct int2
  {
    int x;
    int y;    
  };

template<typename T>
int2 make_int2(T x, T y)
{
  int2 a;
  a.x = x;
  a.y = y;
  return a;
}

struct uint2
  {
    unsigned int x;
    unsigned int y;
  };

  struct float3
  {
  bool operator==(const float3& a) const
    {
      return a.x==x && a.y==y && a.z==z;
    }    
    
    float x;
    float y;
    float z;
  };

template<typename T>
float3 make_float3(T x, T y,T z)
{
  float3 a;
  a.x = x;
  a.y = y;
  a.z = z;
  return a;
}

template<typename T>
float3 make_float3(const T& b)
{
  float3 a;
  a.x = b.x;
  a.y = b.y;
  a.z = b.z;
  return a;
}

float3 operator+(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
void operator-=(float3 &a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

void operator*=(float3 &a, const float3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
float3 operator*(const float3& a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
float3 operator*(float b, const float3& a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

struct float4
  {
    bool operator==(const float4& a) const
    {
      return a.x==x && a.y==y && a.z==z && a.w==w;
    }
    float x;
    float y;
    float z;
    float w;
  };

struct uchar4
  {
    bool operator==(const uchar4& a) const
    {
      return a.x==x && a.y==y && a.z==z && a.w==w;
    }
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
  };
*/



////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

/*
struct int2
  {
    int x;
    int y;    
  };
*/

/*
struct int3
  {
    bool operator<(const int3& a) const
    {
      return x<a.x || (x==a.x && (y<a.y || (y==a.y && z<a.z)));
    }
    
    int x;
    int y;
    int z;    
  };
*/

/*
struct int4
  {
    int x;
    int y;
    int z;
    int w;
  };
*/

typedef unsigned int uint;
/*
struct uint2
  {
    uint2()
    {}

    uint2(unsigned int a)
    {
      x=a;
      y=a;
    }

    bool operator==(const uint2& a) const
    {
      return a.x==x && a.y==y;
    }
    
    unsigned int x;
    unsigned int y;    
  };
*/
/*
struct uint3
  {
    
    
    unsigned int x;
    unsigned int y;
    unsigned int z;
  };
*/
/*
struct uint4
  {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;    
  };
*/
/*
struct float2
  {
    bool operator==(const float2& a) const
    {
      return a.x==x && a.y==y;
    }

    bool operator<(const float2& a) const
    {
      return x<a.x || (x==a.x && y<a.y);
    }
    
    float x;
    float y;
  };
*/
/*
struct double2
  {
    bool operator==(const double2& a) const
    {
      return a.x==x && a.y==y;
    }    
    
    double x;
    double y;
  };
*/

template<typename T>
struct V2
{
  typedef T value_type;
  static const unsigned int size = 2;
  
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  V2() : x(0), y(0)
  {
  }
  
  
  /*
     #ifdef __CUDACC__
  __device__ __host__
#endif
  template<typename U>
  V2(V2<U> v)
  {
    x = v.x;
    y = v.y;
  }
  */

  // template<typename V>
  // #ifdef __CUDACC__
  // __device__ __host__
  // #endif
  // V2(V v) :
  //   x(v.x), y(v.y)
  // {
  // }

  
#ifdef __CUDACC__
  __device__ __host__
  #endif
  V2(T v)
  {
    x = v;
    y = v;
  }

      #ifdef __CUDACC__
  __device__ __host__
  #endif
  V2(const T v0, const T v1) :
    x(v0), y(v1)
  {
  }

  template<typename V>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  V2(const V v) :
    x(v.x), y(v.y)
  {
  }


#ifdef __CUDACC__
  __device__ __host__
#endif
  V2(std::pair<T, T> v)
  {
    x = v.first;
    y = v.second;
  }

  #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator==(const V2& a) const
  {
    return a.x==x && a.y==y;
  }
  
#ifdef __CUDACC__
  __device__ __host__
#endif
  bool eq(const V2& a, T eps) const
  {
    return
      (x > a.x-eps) && (x < a.x+eps) &&
      (y > a.y-eps) && (y < a.y+eps);
  } 

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator!=(const V2& a) const
  {
    return a.x!=x || a.y!=y;
  }

#ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator<(const V2& a) const
  {
    return x<a.x || (x==a.x && y<a.y);
  }

  #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V2<T> operator+(V2<T> lhs, const V2<T>& rhs)
  {
    V2<T> out;
    out.x = lhs.x+rhs.x;
    out.y = lhs.y+rhs.y;
    return out;
  }

#ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V2<T> operator*(T lhs, const V2<T>& rhs)
  {
    V2<T> out;
    out.x = lhs*rhs.x;
    out.y = lhs*rhs.y;
    return out;
  }

  #ifdef __CUDACC__
  __device__ __host__
  #endif
    friend V2<T> operator*(V2<T> lhs, const T& rhs)
  {
    V2<T> out;
    out.x = lhs.x*rhs;
    out.y = lhs.y*rhs;
    return out;
  }
  
#ifdef __CUDACC__
  __device__ __host__
#endif
  friend V2<T> operator*(V2<T> lhs, const V2<T>& rhs)
  {
    V2<T> out;
    out.x = lhs.x*rhs.x;
    out.y = lhs.y*rhs.y;
    return out;
  }

#ifdef __CUDACC__
  __device__ __host__
#endif
  friend inline  void operator/=(V2<T> &a, T b)
  {
    a.x /= b;
    a.y /= b;
  }

   template<typename I>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T& operator[](const I index)
  {
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral for array access");
    assert(!std::is_signed<I>() || index >= 0);
#endif
    
    assert(index < 2);

    if(index)
      return y;
    else
      return x;
    /*
    switch(index)
      {
      case 0:
	return x;
      default:
	return y;
      };
    */
  }
  
  T x;
  T y;
};

//inline __host__ __device__
template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void operator+=(V2<T> &a, V2<T> b)
{
    a.x += b.x;
    a.y += b.y;
}

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void operator-=(V2<T> &a, V2<T> b)
{
    a.x -= b.x;
    a.y -= b.y;
}

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
V2<T> operator-(const V2<T> &a)
{
  return V2<T>(-a.x, -a.y);  
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V2<T> make_V2(T a, T b)
{
  V2<T> o;
  o.x = a;
  o.y = b;
  return o;
}

template<typename T, typename TE>
        #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  V2<T> operator/(V2<T> a, TE b)
{
    return make_V2<T>(a.x / b, a.y / b);
}

template<typename T, typename U>
        #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  V2<T> operator/(V2<T> a, V2<U> b)
{
    return make_V2<T>(a.x / b.x, a.y / b.y);
}

template<typename T>
inline std::ostream& operator<< (std::ostream &out, const V2<T> &v)
{
  out << "(" << v.x << ", " << v.y << ")";
  return out;
}

template<typename T>
struct V3
{
  static const unsigned int size = 3;
  typedef T value_type;

    #ifdef __CUDACC__
  __device__ __host__
  #endif
  V3() : x(0), y(0), z(0) {}

    #ifdef __CUDACC__
  __device__ __host__
  #endif
  V3(T a, T b, T c)
  {
    x=a;
    y=b;
    z=c;
  }

      #ifdef __CUDACC__
  __device__ __host__
  #endif
  V3(T a)
  {
    x=a.x;
    y=a.y;
    z=a.z;
  }

  template<typename U>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  V3(const V3<U>& v3)
  {
    x=v3.x;
    y=v3.y;
    z=v3.z;
  }
  
  template<typename VX>
#ifdef __CUDACC__
  __device__ __host__
#endif
  V3(const VX& v3)
  {
    x=v3.x;
    y=v3.y;
    z=v3.z;
  }

  template<typename U, typename V>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  V3(const V2<U>& v2, V vz)
  {
    x=v2.x;
    y=v2.y;
    z=vz;
  }
 
  
  bool operator==(const V3& a) const
  {
    return a.x==x && a.y==y && a.z==z;
  }

  bool eq(const V3& a, T eps) const
  {
    return
      (x > a.x-eps) && (x < a.x+eps) &&
      (y > a.y-eps) && (y < a.y+eps) &&
      (z > a.z-eps) && (z < a.z+eps);
  } 

  #ifdef __CUDACC__
  __device__ __host__
  #endif  
  V3<T> & operator=(const T& a)
  {
    x = a;
    y = a;
    z = a;
    return *this;
  }

  #ifdef __CUDACC__
  __device__ __host__
  #endif  
  V3<T> operator-() const
  {
    return V3<T>(-x, -y, -z);
  }
  

          #ifdef __CUDACC__
  __device__ __host__
  #endif
    bool operator!=(const V3& a) const
  {
    return a.x!=x || a.y!=y || a.z!=z;
  }

  bool operator<(const V3& a) const
  {
    return x<a.x || (x==a.x && (y<a.y || (y==a.y && z<a.z)));
  }

      #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator*(T lhs, const V3<T>& rhs)
  {
    V3<T> out;
    out.x = lhs*rhs.x;
    out.y = lhs*rhs.y;
    out.z = lhs*rhs.z;
    return out;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator*(const V3<T>& lhs, T rhs)
  {
    V3<T> out;
    out.x = lhs.x*rhs;
    out.y = lhs.y*rhs;
    out.z = lhs.z*rhs;
    return out;
  }

    #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator*(V3<T> lhs, const V3<T>& rhs)
  {
    V3<T> out;
    out.x = lhs.x*rhs.x;
    out.y = lhs.y*rhs.y;
    out.z = lhs.z*rhs.z;
    return out;
  }


        #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator/(T lhs, const V3<T>& rhs)
  {
    V3<T> out;
    out.x = lhs/rhs.x;
    out.y = lhs/rhs.y;
    out.z = lhs/rhs.z;
    return out;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator+(V3<T> lhs, const V3<T>& rhs)
  {
    V3<T> out;
    out.x = lhs.x+rhs.x;
    out.y = lhs.y+rhs.y;
    out.z = lhs.z+rhs.z;
    return out;
  }

  //  template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
friend void operator-=(V3<T> &a, V3<T> b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}


#ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V3<T> operator-(V3<T> lhs, const V3<T>& rhs)
  {
    V3<T> out;
    out.x = lhs.x-rhs.x;
    out.y = lhs.y-rhs.y;
    out.z = lhs.z-rhs.z;
    return out;
  }

    template<typename I>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T& operator[](const I index)
  {
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral for array access");
    assert(!std::is_signed<I>() || index >= 0);
#endif
    
    assert(index < 3);

    switch(index)
      {
      case 0:
	return x;
      case 1:
	return y;
      default:
	return z;
      };
  }

   template<typename I>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  const T& operator[](const I index) const
  {
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral for array access");
    assert(!std::is_signed<I>() || index >= 0);
#endif
    
    assert(index < 3);

    switch(index)
      {
      case 0:
	return x;
      case 1:
	return y;
      default:
	return z;
      };
  }
  
  T x;
  T y;
  T z;
};

  template<typename T>
  #ifdef __CUDACC__
__host__ __device__
#endif
  V3<T> minv(V3<T> a, V3<T> b)
{
  #ifndef __CUDACC__
  using namespace std;
  #endif

  return V3<T>
    (
     min(a.x, b.x),
     min(a.y, b.y),
     min(a.z, b.z)
     );
}

template<typename T>
 #ifdef __CUDACC__
__host__ __device__
#endif
  V3<T> maxv(V3<T> a, V3<T> b)
{
  #ifndef __CUDACC__
  using namespace std;
  #endif

  return V3<T>
    (
     max(a.x, b.x),
     max(a.y, b.y),
     max(a.z, b.z)
     );
}

template<typename T>
 #ifdef __CUDACC__
__host__ __device__
#endif
T maxe(V3<T> a)
{
  #ifndef __CUDACC__
  using namespace std;
  #endif

  return max(a.x, max(a.y, a.z));
}


  template<typename T>
  #ifdef __CUDACC__
__host__ __device__
#endif
void operator+=(V3<T> &a, T b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}


  template<typename T>
  #ifdef __CUDACC__
__host__ __device__
#endif
void operator+=(V3<T> &a, V3<T> b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V3<T> make_V3(T a, T b, T c)
{
  V3<T> o;
  o.x = a;
  o.y = b;
  o.z = c;
  return o;
}

template<typename T>
inline V3<T> make_V3(T a)
{
  V3<T> o;
  o.x = a;
  o.y = a;
  o.z = a;
  return o;
}

template<typename T_OUT, typename T>
inline V3<T_OUT> make(V3<T> in)
{
  V3<T_OUT> o;
  o.x = in.x;
  o.y = in.y;
  o.z = in.z;
  return o;
}

template<typename T>
inline std::ostream& operator<< (std::ostream &out, const V3<T> &v)
{
  out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return out;
}

/*
struct float3
  {
    bool operator==(const float3& a) const
    {
      return a.x==x && a.y==y && a.z==z;
    }    
    
    float x;
    float y;
    float z;
  };
*/

template<typename T>
struct V4
{
  static const unsigned int size = 4;
  typedef T value_type;
  
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  V4() : x(0), y(0), z(0), w(0) {}

  
#ifdef __CUDACC__
  __device__ __host__
  #endif
  V4(T a, T b, T c, T d)
  {
    x=a;
    y=b;
    z=c;
    w=d;
  }

#ifdef __CUDACC__
  __device__ __host__
  #endif
  V4(V3<T> a, T d)
  {
    x=a.x;
    y=a.y;
    z=a.z;
    w=d;
  }

  template<typename U>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  V4(V4<U> a)
  {
    x=a.x;
    y=a.y;
    z=a.z;
    w=a.w;
  }

  
        #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator==(const V4& a) const
  {
    return a.x==x && a.y==y && a.z==z && a.w==w;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator!=(const V4& a) const
  {
    return a.x!=x || a.y!=y || a.z!=z || a.w != w;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  bool operator<(const V4& a) const
  {
    return x<a.x || (x==a.x && (y<a.y || (y==a.y && (z<a.z || (z==a.z && w<a.w)))));
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V4<T> operator+(V4<T> lhs, const V4<T>& rhs)
  {
    V4<T> out;
    out.x = lhs.x+rhs.x;
    out.y = lhs.y+rhs.y;
    out.z = lhs.z+rhs.z;
    out.w = lhs.w+rhs.w;
    return out;
  }

      #ifdef __CUDACC__
  __device__ __host__
  #endif
friend void operator+=(V4<T> &a, V4<T> b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

        #ifdef __CUDACC__
  __device__ __host__
  #endif
  friend V4<T> operator-(V4<T> lhs, const V4<T>& rhs)
  {
    V4<T> out;
    out.x = lhs.x-rhs.x;
    out.y = lhs.y-rhs.y;
    out.z = lhs.z-rhs.z;
    out.w = lhs.w-rhs.w;
    return out;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
friend V4<T> operator*(V4<T> lhs, const T& rhs)
  {
    V4<T> out;
    out.x = lhs.x*rhs;
    out.y = lhs.y*rhs;
    out.z = lhs.z*rhs;
    out.w = lhs.w*rhs;
    return out;
  }

        #ifdef __CUDACC__
  __device__ __host__
  #endif
friend V4<T> operator*(const T& rhs, V4<T> lhs)
  {
    V4<T> out;
    out.x = lhs.x*rhs;
    out.y = lhs.y*rhs;
    out.z = lhs.z*rhs;
    out.w = lhs.w*rhs;
    return out;
  }
  
#ifdef __CUDACC__
  __device__ __host__
#endif
  friend V4<T> operator*(V4<T> lhs, const V4<T>& rhs)
  {
    V4<T> out;
    out.x = lhs.x*rhs.x;
    out.y = lhs.y*rhs.y;
    out.z = lhs.z*rhs.z;
    out.w = lhs.w*rhs.w;
    return out;
  }


      #ifdef __CUDACC__
  __device__ __host__
  #endif
friend inline  void operator/=(V4<T> &a, T b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

      #ifdef __CUDACC__
  __device__ __host__
  #endif
friend inline  void operator*=(V4<T> &a, T b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

  template<typename I>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T& operator[](const I index)
  {
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral for array access");
    assert(!std::is_signed<I>() || index >= 0);
#endif
    
    assert(index < 4);

    switch(index)
      {
      case 0:
	return x;
      case 1:
	return y;
      case 2:
	return z;
      case 3:
      default:
	return w; 
      };
  }

    template<typename I>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  const T& operator[](const I index) const
  {
    #ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral for array access");
    #endif

#ifndef __CUDACC__
    assert(!std::is_signed<I>() || index >= 0);
#endif
    assert(index < 4);

    switch(index)
      {
      case 0:
	return x;
      case 1:
	return y;
      case 2:
	return z;
      case 3:
      default:
	return w; 
      };
  }


  void toArray(T* o)
  {
    o[0] = x;
    o[1] = y;
    o[2] = z;
    o[3] = w;
  }
  
  T x;
  T y;
  T z;
  T w;
};

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V4<T> make_V4(T a, T b, T c, T d)
{
  V4<T> o;
  o.x = a;
  o.y = b;
  o.z = c;
  o.w = d;
  return o;
}

template<typename T>
inline std::ostream& operator<< (std::ostream &out, const V4<T> &v)
{
  out << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return out;
}



template<typename T>
inline  void operator*=(V2<T> &a, T b)
{
    a.x *= b;
    a.y *= b;
}

/*
struct float4
  {
    bool operator==(const float4& a) const
    {
      return a.x==x && a.y==y && a.z==z && a.w==w;
    }
    float x;
    float y;
    float z;
    float w;
  };

struct uchar4
  {
    bool operator==(const uchar4& a) const
    {
      return a.x==x && a.y==y && a.z==z && a.w==w;
    }
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
  };
*/


//#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////
/*
template<typename T, typename IN>
  #ifdef __NVCC__
   __device__ __host__
#endif
T make_vec(const IN& a)
{
  return T(a);
}
*/

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
inline  V2<T> operator-(V2<T> &a)
{
  V2<T> o;
  o.x = -a.x;
  o.y = -a.y;
  return o;
  //return make_vec<V2<T>>(-a.x, -a.y);
}

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
inline V2<T> operator-(V2<T> a,  V2<T> b)
{
  return make_V2<T>(a.x - b.x, a.y - b.y);
}

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
inline V3<T> operator-(V3<T> a,  V3<T> b)
{
  return make_V3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V3<T> operator/(V3<T> a, V3<T> b)
{
    return make_V3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V3<T> operator/(V3<T> a, double b)
{
    return make_V3<T>(a.x / b, a.y / b, a.z / b);
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  void operator/=(V3<T> &a, V3<T> b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}


template<typename T0, typename T1>
        #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  void operator/=(V3<T0> &a, const T1& b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

template<typename T0, typename T1>
        #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  void operator*=(V3<T0> &a, V3<T1> b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}


template<typename T0, typename T1>
        #ifdef __CUDACC__
  __device__ __host__
  #endif
inline  void operator*=(V3<T0> &a, const T1& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

#ifdef M_VEC
typedef V2<float> m_float2;
typedef V2<double> m_double2;
typedef V2<int> m_int2;
typedef V2<unsigned int> m_uint2;

typedef V3<float> m_float3;
typedef V3<unsigned char> m_uchar3;
typedef V3<double> m_double3;
typedef V3<int> m_int3;
typedef V3<unsigned int> m_uint3;

typedef V4<float> m_float4;
typedef V4<double> m_double4;
typedef V4<int> m_int4;
typedef V4<unsigned int> m_uint4;
typedef V4<unsigned char> m_uchar4;

template<typename T, typename U>
  void m_assign(T& out, const U& in)
{
  out = in;
}

template<typename T, typename U>
void m_assign(T& out, const V2<U>& in)
{
  out.x = in.x;
  out.y = in.y;
}

template<typename T, typename U>
void m_assign(T& out, const V3<U>& in)
{
  out.x = in.x;
  out.y = in.y;
  out.z = in.z;
}


#else


typedef V2<float> float2;
typedef V2<double> double2;
typedef V2<int> int2;
typedef V2<unsigned int> uint2;

typedef V3<float> float3;
typedef V3<unsigned char> uchar3;
typedef V3<double> double3;
typedef V3<int> int3;
typedef V3<unsigned int> uint3;

typedef V4<float> float4;
typedef V4<double> double4;
typedef V4<int> int4;
typedef V4<unsigned int> uint4;
typedef V4<unsigned char> uchar4;

/*
template<typename T, typename E>
void mvec_init(T& v, E e)
{
  v = e;
}
*/

uchar4 make_uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d)
{
  return make_V4<unsigned char>(a,b,c,d);
}

inline float2 make_float2(float a, float b)
{
  float2 f;
  f.x = a;
  f.y = b;
  return f;
}

inline double2 make_double2(double a, double b)
{
  double2 f;
  f.x = a;
  f.y = b;
  return f;
}


inline int2 make_int2(int a, int b)
{
  int2 f;
  f.x = a;
  f.y = b;
  return f;
}

inline  uint2 make_uint2(uint a, uint b)
{  
    
  uint2 f;
  f.x = a;
  f.y = b;
  return f;
}

inline float3 make_float3(float a, float b, float c)
{
  float3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

inline uchar3 make_uchar3(unsigned char a, unsigned char b, unsigned char c)
{
  uchar3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

inline double3 make_double3(double a, double b, double c)
{
  double3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

inline int3 make_int3(int a, int b, int c)
{
  int3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

inline  uint3 make_uint3(uint a, uint b, uint c)
{
  uint3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

inline float4 make_float4(float a, float b, float c, float d)
{
  float4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;
}

inline int4 make_int4(int a, int b, int c, int d)
{
  int4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;
}

inline  uint4 make_uint4(uint a, uint b, uint c, uint d)
{
  uint4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;
}

inline  double4 make_double4(double a, double b, double c, double d)
{
  double4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;
}

inline  float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline  float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline  float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline  float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline  int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline  int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline  int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline  int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline  uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline  uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline  uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline  float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline  float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline  float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline  float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline  float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline  float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline  int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline  int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline  int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline  int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline  int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline  uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline  uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline  uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline  uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline  uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline  float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline  float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline  float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline  float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline  float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline  int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline  int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline  int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline  int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline  int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline  uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline  uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline  uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline  uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

/*
inline  int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
*/

inline  float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline  int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline  float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline  int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////
/*
inline  float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
*/
/*
inline  void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
*/
inline  float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline  float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
/*
inline  void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}
*/

inline  int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
/*
inline  void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
*/
inline  int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline  int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
/*
inline  void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}
*/

inline  uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
/*
inline  void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
*/
inline  uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline  uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
/*
inline  void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}
*/

template<typename T>
inline  V3<T> operator+(V3<T> a, V3<T> b)
{
    return make_V3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}



template<typename T>
inline  void operator+=(V3<T> &a, V3<T> b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline  float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline  void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline  int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline  void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline  int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline  void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline  uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline  void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline  uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline  void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline  int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline  uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline  float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline  float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline  float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline  float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline  void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline  int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline  int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline  int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline  void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline  uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline  uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline  uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline  void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////


inline  void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline  float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline  float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline  void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline  int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline  void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline  int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline  int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline  void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline  uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline  void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline  uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline  uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline  void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}



/*
inline  float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
*/

inline  void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline  float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline  float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline  void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline  int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline  void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline  int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline  int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline  void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline  uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline  void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline  uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline  uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline  void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline  float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline  void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline  float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline  void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline  int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline  void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline  int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline  int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline  void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline  uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline  void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline  uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline  uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline  void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline  V2<T> operator*(V2<T> a, V2<T> b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

template<typename T>
inline  void operator*=(V2<T> &a, V2<T> b)
{
  a.x *= b.x;
  a.y *= b.y;
}

template<typename T, typename U>
inline V2<T> operator*(V2<T> a, U b)
{
V2<T> o;
  o.x = b * a.x;
  o.y = b * a.y;
  return o;
}

template<typename T, typename U>
inline V4<T> operator*(V4<T> a, U b)
{
V4<T> o;
  o.x = b * a.x;
  o.y = b * a.y;
  o.z = b * a.z;
  o.w = b * a.w;
  return o;
}

template<typename T>
inline std::ostream& operator<< (std::ostream &out, const V2<T> &v)
{
  out << "(" << v.x << ", " << v.y << ")";
  return out;
}

template<typename T, typename U>
inline V2<T> operator*(U b, V2<T> a)
{
  V2<T> o;
  o.x = b * a.x;
  o.y = b * a.y;
  return o;
}


inline  int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline  void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline  int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline  int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline  void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline  uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline  void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline  uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline  uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline  void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}
/*
inline  float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
*/
inline  void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline  float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
/*
inline  float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
*/
inline  void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline  int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

template<typename T>
inline  V3<T> operator*(V3<T> a, T b)
{
    return make_V3<T>(a.x * b, a.y * b, a.z * b);
}

template<typename T>
inline V3<T> operator*(T b, V3<T> a)
{
    return make_V3<T>(b * a.x, b * a.y, b * a.z);
}

inline  void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline  uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline  uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline  uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline  void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline  float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline  float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline  float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline  void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline  int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline  int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline  int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline  void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline  uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline  uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline  uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline  void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V2<T> operator/(V2<T> a, V2<T> b)
{
    return make_V2<T>(a.x / b.x, a.y / b.y);
}


inline  void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}

template<typename T, typename U>
inline V2<T> operator/(V2<T> a, U b)
{
    return make_V2<T>(a.x / b, a.y / b);
}


template<typename T>
inline T dot(V2<T> a, V2<T> b)
{
  return a.x*b.x+a.y*b.y;
}

template<typename T>
inline T dist2(V2<T> a, V2<T> b)
{
  const auto tmp = a-b;
  return dot(tmp,tmp);
}


template<typename T>
inline V2<T> operator+(V2<T> a, V2<T> b)
{
  return make_V2<T>(a.x+b.x, a.y+b.y);
}

template<typename T>
inline V4<T> operator+(V4<T> a, V4<T> b)
{
  return make_V4<T>(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

template<typename T>
inline  void operator+=(V2<T> &a, V2<T> b)
{
    a.x += b.x;
    a.y += b.y;
}

inline  void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline  float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}


inline  float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline  float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline  void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline  float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline  void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline  float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline   float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline  float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline   float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline  int2 min(int2 a, int2 b)
{
  return make_int2(std::min(a.x,b.x), std::min(a.y,b.y));
}
inline  int3 min(int3 a, int3 b)
{
  return make_int3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}
inline  int4 min(int4 a, int4 b)
{
  return make_int4(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z), std::min(a.w,b.w));
}

inline  uint2 min(uint2 a, uint2 b)
{
    return make_uint2(std::min(a.x,b.x), std::min(a.y,b.y));
}
inline  uint3 min(uint3 a, uint3 b)
{
    return make_uint3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}
inline  uint4 min(uint4 a, uint4 b)
{
    return make_uint4(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z), std::min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline  float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline  float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline  float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline  int2 max(int2 a, int2 b)
{
    return make_int2(std::max(a.x,b.x), std::max(a.y,b.y));
}
inline  int3 max(int3 a, int3 b)
{
    return make_int3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}
inline  int4 max(int4 a, int4 b)
{
    return make_int4(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z), std::max(a.w,b.w));
}

inline  uint2 max(uint2 a, uint2 b)
{
    return make_uint2(std::max(a.x,b.x), std::max(a.y,b.y));
}
inline  uint3 max(uint3 a, uint3 b)
{
    return make_uint3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}
inline  uint4 max(uint4 a, uint4 b)
{
    return make_uint4(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z), std::max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline  float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline  float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline  float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline  float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline  float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline  int clamp(int f, int a, int b)
{
  return std::max(a, std::min(f, b));
}
inline  uint clamp(uint f, uint a, uint b)
{
  return std::max(a, std::min(f, b));
}

inline  float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline  float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline  float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline  float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline  float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline  float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline  int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline  int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline  int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline  int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline  int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline  int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline  uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline  uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline  uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline  uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline  uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline  uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline  float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

template<typename T>
inline  float dot(V3<T> a, V3<T> b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline  float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline  int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline  int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline  int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline  uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline  uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline  uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline  float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline  float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline  float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////


/*
inline  float2 normalize(float2 v)
{
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}
inline  float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline  float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
*/

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline  float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline  float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline  float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline  float fracf(float v)
{
    return v - floorf(v);
}
inline  float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline  float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline  float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline  float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline  float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline  float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline  float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline  float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline  float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline  int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline  int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline  int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline  float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline  float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline  float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline  float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline  float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline  float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}




inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}

#endif

template<typename T>
    #ifdef __CUDACC__
  __device__ __host__
  #endif
inline T dot(V2<T> a, V2<T> b)
{
  return a.x*b.x+a.y*b.y;
}

template<typename T>
    #ifdef __CUDACC__
  __device__ __host__
  #endif
inline T dot(V3<T> a, V3<T> b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

template<typename T>
    #ifdef __CUDACC__
  __device__ __host__
  #endif
inline T dot(V4<T> a, V4<T> b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
}

template<typename T>
inline T dot(T a, T b)
{
  return a*b;
}

template<typename T>
    #ifdef __CUDACC__
  __device__ __host__
  #endif
inline V3<T> cross(V3<T> v0, V3<T> v1)
{
  V3<T> result;
  result[0] = v0[1] * v1[2] - v0[2] * v1[1];
  result[1] = v0[2] * v1[0] - v0[0] * v1[2];
  result[2] = v0[0] * v1[1] - v0[1] * v1[0];;
  return result;
}


template<typename T>
inline T length2(V2<T> a)
{
  return dot(a,a);
}

template<typename T>
inline T length2(V3<T> a)
{
  return dot(a,a);
}

template<typename T>
inline T length(T a)
{
  return sqrtf(a*a);
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline T length(V2<T> a)
{
  return sqrtf(dot(a,a));
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
inline T length(V3<T> a)
{
  return sqrtf(dot(a,a));
}

template<typename T>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
T normalize(T v)
{
  return v/length(v);
}

// template<typename T>
// uint32_t mvec_size()
// {
//   return 1;
// }


template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(V3<float>& v, E e)
{
  v.x = e;
  v.y=e;
  v.z=e;
}

template<typename T, typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(V2<T>& v, E e)
{
  v.x = e;
  v.y=e;
}


template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(V3<double>& v, E e)
{
  v.x = e;
  v.y=e;
  v.z=e;
}

template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(unsigned char& v, E e)
{
  v = e;
}

template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(uint16_t& v, E e)
{
  v = e;
}

template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(double& v, E e)
{
  v = e;
}

template<typename E>
#ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(float& v, E e)
{
  v = e;
}


template<typename T_out, typename T_in>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
T_out make_vec(const T_in& v)
{
  T_out out = v;
  return out;
}

template<typename T>
#ifdef __CUDACC__
  __device__ __host__
  #endif
V2<T> make_ortho(const V2<T>& v)
{
  V2<T> out;
  out.y = v.x;
  out.x = -v.y;
  return out;
}

template<typename T_out, typename T_in>
#ifdef __CUDACC__
  __device__ __host__
  #endif
V2<T_out> convert(const V2<T_in> v)
{
  return V2<T_out>(v.x, v.y);
}

template<typename T_out, typename T_in>
#ifdef __CUDACC__
  __device__ __host__
  #endif
V3<T_out> convert(const V3<T_in> v)
{
  return V3<T_out>(v.x, v.y, v.z);
}

template<typename T_out, typename T_in>
#ifdef __CUDACC__
  __device__ __host__
  #endif
V4<T_out> convert(const V4<T_in> v)
{
  return V3<T_out>(v.x, v.y, v.z, v.w);
}

template<typename T_out, typename T_in>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T_out make_vec(const T_in& v0, const T_in& v1)
{
  T_out out;
  out.x = v0;
  out.y = v1;
  return out;
}


template<typename T_out, typename T_in0, typename T_in1, typename T_in2>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  T_out make_vec(const T_in0& v0, const T_in1& v1, const T_in2& v2)
{
  T_out out;
  out.x = v0;
  out.y = v1;
  out.z = v2;
  return out;
}

template<typename T_out, typename T_in>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
  T_out make_vec(const T_in& v0, const T_in& v1, const T_in& v2, const T_in& v3)
{
  T_out out;
  out.x = v0;
  out.y = v1;
  out.z = v2;
  out.w = v3;
  return out;
}

//
// DEFINE FUNCTIONS FOR CUDA VECTORS
//
#if defined(__VECTOR_TYPES_H__)
template<typename E>
      #ifdef __CUDACC__
  __device__ __host__
  #endif
void mvec_init(float3& v, E e)
{
  v.x = e;
  v.y=e;
  v.z=e;
}
/*

#if defined(__VECTOR_TYPES_H__)
      #ifdef __CUDACC__
  __device__ __host__
  #endif
bool operator!=(const float3& a, const float3& b)
  {
    return a.x!=b.x || a.y!=b.y || a.z!=b.z;
  }

#endif
*/

#endif

#endif //__VEC__
