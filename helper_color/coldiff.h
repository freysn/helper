#ifndef __COLDIFF__
#define __COLDIFF__


namespace coldiff
{
  template<typename T, typename U>
#ifdef __CUDACC__
    __device__ __host__
#endif
    void diff(U& diff, const T& a, const T& b)
    {
      diff = length(a-b);
    }

  template<typename T, typename U>
#ifdef __CUDACC__
    __device__ __host__
#endif
    void diffPremultiplied(U& diff, const T& a, const T& b)
    {
      T ap = a;
      ap.x *= ap.w;
      ap.y *= ap.w;
      ap.z *= ap.w;
  

      T bp = b;
      bp.x *= bp.w;
      bp.y *= bp.w;
      bp.z *= bp.w;
            
      diff = length(ap-bp);
    }


    template<typename T, typename U>
#ifdef __CUDACC__
    __device__ __host__
#endif
      void diffMagic(U& diff, const T& a, const T& b)
    {
      //return diff(a,b);
      //diff = diffPremultiplied(a,b);
      
      T pa = a.w*a;
      T pb = b.w*b;

      T v;      
      v.x = pa.x-pb.x;
      v.y = pa.y-pb.y;
      v.z = pa.z-pb.z;

      diff.x = sqrtf(v.x*v.x+
                     v.y*v.y+
                     v.z*v.z);
      
      //diff.y = fabs(a.w-b.w);

      if(max(a.w, b.w) <= 0.00000001)
        diff.y = 1;
      else
        diff.y = min(a.w, b.w)/max(a.w, b.w);
    }

    template<typename T, typename U>
#ifdef __CUDACC__
    __device__ __host__
#endif
      void diffMagic2(U& diff, const T& a, const T& b)
    {
      //return diff(a,b);
      //diff = diffPremultiplied(a,b);
      
      T pa = a.w*a;
      T pb = b.w*b;

      T v;      
      v.x = pa.x-pb.x;
      v.y = pa.y-pb.y;
      v.z = pa.z-pb.z;

      diff.x = sqrtf(v.x*v.x+
                     v.y*v.y+
                     v.z*v.z);
      
      //diff.y = fabs(a.w-b.w);

      diff.y = fabs(a.w-b.w);      
    }
}

#endif //__COLOR_DIFFERENCE__
