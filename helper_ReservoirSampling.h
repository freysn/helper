#ifndef __HELPER_RESERVOIR_SAMPLING__
#define __HELPER_RESERVOIR_SAMPLING__

#include "helper_util.h"

namespace helper
{
  // https://en.wikipedia.org/wiki/Reservoir_sampling

//   (*
//   S has items to sample, R will contain the result
//  *)
// ReservoirSample(S[1..n], R[1..k])
//   // fill the reservoir array
//   for i = 1 to k
//       R[i] := S[i]

//   // replace elements with gradually decreasing probability
//   for i = k+1 to n
//     j := random(1, i)   // important: inclusive range
//     if j <= k
//         R[j] := S[i]

  template<uint64_t _n, typename T, typename RNG>
  class helper_ReservoirSampling
  {
  public:
    #ifdef __CUDACC__
    __host__ __device__
#endif
    helper_ReservoirSampling(T* d, RNG rng) :
      _d(d), _cnt(0), _rng(rng)
      {
      }

#ifdef __CUDACC__
    __host__ __device__
#endif
    void operator()(const T p)
    {
      if(_cnt < _n)
	{
	  _d[_cnt] = p;
	}
      else
	{
	  // generate random integer in [0, _cnt-1]
	  const auto g = _rng(0, _cnt-1);
	  assert(g >= 0 && g < _cnt);
	  
	  if(g < _n)
	    _d[g] = p;
	}

      _cnt++;
    }


    #ifdef __CUDACC__
    __host__ __device__
#endif
    auto getNOccupiedSlots() const
    {
      return helper::min(_cnt, _n);
    }
    

    T*_d;
    uint64_t _cnt;
    RNG _rng;
  };
  
}

#endif //__HELPER_RESERVOIR_SAMPLING__
