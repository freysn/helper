#ifndef __HELPER_RAY__
#define __HELPER_RAY__

namespace helper
{
template<typename F3, typename F>
#ifdef __CUDACC__
  __host__ __device__
#endif    
    int intersectBox(F3 ro, F3 rd, F3 boxmin, F3 boxmax, F *tnear, F *tfar)
  {
    
    /*
    auto MIN3 = [](const F3& a, const F3& b)
      {
	F3 out;
	out.x = std::min(a.x, b.x);
	out.y = std::min(a.y, b.y);
	out.z = std::min(a.z, b.z);
	return out;
      };
    auto MAX3 = [](const F3& a, const F3& b)
      {
	F3 out;
	out.x = std::max(a.x, b.x);
	out.y = std::max(a.y, b.y);
	out.z = std::max(a.z, b.z);
	return out;
      };
    */
    auto MIN1 = [](const float& a, const float& b)
      {
#ifndef __CUDACC__
	using namespace std;
	#endif
	return min(a,b);
      };

    auto MAX1 = [](const float& a, const float& b)
      {
	#ifndef __CUDACC__
	using namespace std;
	#endif
	return max(a,b);
      };

      auto MIN3 = [&MIN1](F3 a, F3 b)
      {
	F3 out;
	out.x = MIN1(a.x, b.x);
	out.y = MIN1(a.y, b.y);
	out.z = MIN1(a.z, b.z);
	return out;
      };

      auto MAX3 = [&MAX1](F3 a, F3 b)
      {
	F3 out;
	out.x = MAX1(a.x, b.x);
	out.y = MAX1(a.y, b.y);
	out.z = MAX1(a.z, b.z);
	return out;
      };
      
    // compute intersection of ray with all six bbox planes
    //F3 invR = make_F3(1.0f) / rd;
      F3 invR;
      invR.x = 1. / rd.x;
      invR.y = 1. / rd.y;
      invR.z = 1. / rd.z;
      
      F3 tbot;
      tbot.x= invR.x * (boxmin.x - ro.x);
      tbot.y= invR.y * (boxmin.y - ro.y);
      tbot.z= invR.z * (boxmin.z - ro.z);
      
      F3 ttop;// = invR * (boxmax - ro);
      ttop.x = invR.x * (boxmax.x - ro.x);
      ttop.y = invR.y * (boxmax.y - ro.y);
      ttop.z = invR.z * (boxmax.z - ro.z);
      
    // re-order intersections to find smallest and largest on each axis
    F3 tmin = MIN3(ttop, tbot);
    F3 tmax = MAX3(ttop, tbot);

    // find the largest tmin and the smallest tmax
    F largest_tmin = MAX1(MAX1(tmin.x, tmin.y), MAX1(tmin.x, tmin.z));
    F smallest_tmax = MIN1(MIN1(tmax.x, tmax.y), MIN1(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
  }

  template<typename Ray, typename F3, typename F>
    int intersectBox(Ray r, F3 boxmin, F3 boxmax, F *tnear, F *tfar)
  {
    return intersectBox(r.o, r.d, boxmin, boxmax, tnear, tfar);
  }

}

#endif //__HELPER_RAY__
