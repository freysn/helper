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

  

  template<class RandomIt, class RandomFunc>
  void mrandom_shuffle(RandomIt first, RandomIt last, RandomFunc&& r)
  {
    typename std::iterator_traits<RandomIt>::difference_type i, n;
    n = last - first;
    for (i = n-1; i > 0; --i) {
      using std::swap;
      swap(first[i], first[r()%(i+1)]);
    }
  }
};

#endif //__HELPER_RANDOM__
