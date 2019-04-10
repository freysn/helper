#ifndef __CM_MAP__
#define __CM_MAP__

#include <cassert>
#include <cmath>
#include "helper/helper_util.h"

template<typename F, typename C>
C cm_map_norm(const F v, const std::vector<C>& cm)
{
  //assert(v >= 0. && v<=1.);
  const F under_one = std::nextafter(static_cast<F>(1), static_cast<F>(0));
    
  const auto idx = static_cast<size_t>(helper::clamp(v, static_cast<F>(0), under_one)*cm.size());
  assert(idx<cm.size());
  return cm[idx];
}

template<typename F2, typename C>
C cm_bi_map_norm(const F2 v, const std::vector<C>& cm, const size_t nElemsPerDim)
{
  using F = decltype(v.x);
  //assert(v >= 0. && v<=1.);
  const F under_one = std::nextafter(static_cast<F>(1), static_cast<F>(0));
    
  const auto idx0 = static_cast<size_t>(helper::clamp(v.x, static_cast<F>(0), under_one)*nElemsPerDim);
  const auto idx1 = static_cast<size_t>(helper::clamp(v.y, static_cast<F>(0), under_one)*nElemsPerDim);
  
  const auto idx = idx0+nElemsPerDim*idx1;
  assert(idx<cm.size());
  return cm[idx];
}

template<typename IN_it, typename OUT_it, typename C, typename F>
void cm_map_norm(const IN_it vb, const IN_it ve, OUT_it cb, const std::vector<C>& cm, F ref)
{
  for(auto it=vb; it != ve; it++, cb++)
    *cb = cm_map_norm((*it)/ref, cm);
}


template<typename IN_it, typename C>
auto cm_map_norm(const IN_it vb, const IN_it ve, const std::vector<C>& cm)
{
  std::vector<C> rslt(ve-vb);
  auto cb = rslt.begin();
  for(auto it=vb; it != ve; it++, cb++)
    *cb = cm_map_norm(*it, cm);
  return rslt;
}


template<typename V4, typename IN_it, typename C, typename IN_jt>
auto cm_map_norm_RGBA(const IN_it vb, const IN_it ve, const std::vector<C>& cm, const IN_jt ab)
{
  std::vector<V4> rslt(ve-vb);
  auto cb = rslt.begin();
  auto jt = ab;
  for(auto it=vb; it != ve; it++, cb++, jt++)
    {
      const auto tmp = cm_map_norm(*it, cm);
      cb->x = tmp.x;
      cb->y = tmp.y;
      cb->z = tmp.z;
      cb->w = *jt;
    }
  return rslt;
}




#endif
