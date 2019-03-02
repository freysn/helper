#ifndef __HELPER_RANDOM__
#define __HELPER_RANDOM__

#include <random>

namespace helper
{
  template<typename T, typename RNG>
  size_t pickElementFromDistribution(const std::vector<T>& dist, RNG& rng)
  {
    assert(!dist.empty());
    
    std::vector<T> distSum(dist.size());
    std::partial_sum(dist.begin(), dist.end(), distSum.begin());

    std::uniform_real_distribution<> dis(0, distSum.back());
    // first element which does not compare less than val
    auto it = std::lower_bound(distSum.begin(), distSum.end(), dis(rng));
    assert(it != distSum.end());
    const size_t idx = it-distSum.begin();
    return idx;
  }
};

#endif //__HELPER_RANDOM__
