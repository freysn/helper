#ifndef __SHEPARD__
#define __SHEPARD__

#include <vector>
#include "splitStr.h"
#include <cassert>
#include "helper_idx.h"
#include "readCorners.h"

template<typename T, typename I3, typename P>
std::vector<T>
  shepard(const std::vector<T>& in, const I3& dim,
          const std::vector<P>& points0,
          const std::vector<P>& points1)
{
  assert(!points0.empty());
  std::vector<T> out(in.size());

  assert(points0.size()==points1.size());

  const double maxDim = (double)std::max(dim.x, std::max(dim.y, dim.z));
  for(size_t i=0; i<in.size(); i++)
    {
      auto v = i2iii(i, dim);
      const double3 vf = make_double3(v.x, v.y, v.z);
      
      double sumd = 0.;
      double3 sumv = make_double3(0., 0., 0.);
      
      for(size_t j=0; j<points0.size();j++)
        {
          const auto p = points0[j];
          const double3 pf = make_double3(p.x, p.y, p.z);
          double d2 = length2((pf-vf)/maxDim);
          d2 = std::max(d2, 1e-20);
          double invd2 = 1./d2;
          
          sumv += invd2*(pf-make_double3(points1[j].x,points1[j].y,points1[j].z));
          sumd += invd2;
        }
      const double3 pf = vf+sumv/sumd;
      I3 pi;
      pi.x = pf.x+0.5;
      pi.y = pf.y+0.5;
      pi.z = pf.z+0.5;

      const auto value = in[iii2i_clamp(pi, dim)];
      
      std::cout << "lookup pos for " << v << " - " << pi << "| " << (int)value << std::endl;
      
      out[i] = value;
    }
  return out;
}


#endif //__SHEPARD__
