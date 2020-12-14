#ifndef __HELPER_HASH__
#define __HELPER_HASH__

#include <functional>
#include <utility>

namespace helper
{
  template <class T>
  inline void hash_combine(std::size_t& seed, const T& v)
  {
    //std::hash<T> hasher;
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }

  template <class T>
  inline void hash_combine(std::size_t& seed, const V4<T>& v)
  {
    hash_combine(seed, v.x);
    hash_combine(seed, v.y);
    hash_combine(seed, v.z);
    hash_combine(seed, v.w);
  }

  template <class T>
  inline void hash_combine(std::size_t& seed, const V3<T>& v)
  {
    hash_combine(seed, v.x);
    hash_combine(seed, v.y);
    hash_combine(seed, v.z);   
  }



  template <class T_it>
  inline size_t hash_range(T_it b, T_it e, size_t seed=1337)
  {
    //size_t seed=1337;
    // for_each guarantees order

    typedef decltype(*b) T;
    std::for_each(b, e, [&seed](T a) {hash_combine(seed, a);});

    return seed;
  }
  
};

#endif //__HELPER_HASH__
