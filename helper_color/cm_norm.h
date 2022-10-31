#ifndef __CM_NORM__
#define __CM_NORM__

#include "helper/helper_util.h"

template<typename T>
T cm_norm(const T m, const double targetBrightness)
{
  T out = m;
  for(auto & e : out)
    {
      const double b = 0.299 * e[0] + 0.587 * e[1] + 0.114 * e[2];
      assert(b >= helper::eps<double>());
      const double fac = targetBrightness/b;
      e[0]*= fac;
      e[1]*= fac;
      e[2]*= fac;
    }
  return out;
}


#endif //__CM_NORM__
