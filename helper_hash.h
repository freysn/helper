#ifndef __HELPER_HASH__
#define __HELPER_HASH__

#include <functional>
#include <utility>
#include "helper/helper_omp.h"

namespace helper
{
  inline void hash_combine_seeds(std::size_t& seed, const size_t& seed2)
  {
    //std::hash<T> hasher;
    seed ^= seed2 + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }
  
  template <class T>
  inline void hash_combine(std::size_t& seed, const T& v)
  {
    //std::hash<T> hasher;
    //seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine_seeds(seed, static_cast<size_t>(std::hash<T>()(v)));
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

  template <class T>
  inline void hash_combine(std::size_t& seed, const V2<T>& v)
  {
    hash_combine(seed, v.x);
    hash_combine(seed, v.y);
  }




  template <class T_it>
  inline void hash_combine_range(std::size_t& seed, T_it b, T_it e)
  {    
    typedef decltype(*b) T;
    std::for_each(b, e, [&seed](T a) {hash_combine(seed, a);});    
  }

  template <class T_it>
  inline size_t hash_range(T_it b, T_it e)    
  {
    size_t seed=1337;
    // for_each guarantees order
    
    hash_combine_range(seed, b,e);

    return seed;
  }

  
  template <class T_it>
  inline size_t hash_range_parallel(T_it b, T_it e)
  {
    const size_t nElems = e-b;
    
    const size_t nChunks = std::min(static_cast<size_t>(helper::getMaxNThreadsOMP()*8), nElems);
    const size_t chunkSize = iDivUp(nElems, nChunks);
    assert(chunkSize > 0);
    
    std::vector<size_t> hashes(nChunks);
    
#pragma omp parallel for
    for(size_t i=0; i<nChunks; i++)
      hashes[i] = hash_range(std::min(b+i*chunkSize,e), std::min(b+(i+1)*chunkSize,e));
    
    size_t seed=1337;
    for(const auto & e : hashes)
      hash_combine_seeds(seed, e);
    return seed;
  }
  
};

#endif //__HELPER_HASH__
