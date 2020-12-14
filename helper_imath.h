#ifndef __HELPER_IMATH__
#define __HELPER_IMATH__

#include <iostream>

namespace helper
{
  bool isPow2(uint32_t n)
  {
    return n !=0 && ((n&(n-1)) == 0);
  }

  uint32_t nextPow2(uint32_t x)
  {
    x--;
    x |= x>>1;
    x |= x>>2;
    x |= x>>4;
    x |= x>>8;
    x |= x>>16;
    x++;
    return x;
  }

  // uint64_t nextPow2(uint64_t x)
  // {
  //   x |= x>>1;
  //   x |= x>>2;
  //   x |= x>>4;
  //   x |= x>>8;
  //   x |= x>>16;
  //   x |= x>>32;
  //   return x;
  // }

  bool isPow4(uint32_t n)
  {
    return isPow2(n) && !(n & 0xAAAAAAAA);
  }

  uint32_t ilog2(const uint32_t n)
  {
    const auto nLeading0Bits=__builtin_clz(n);
    const uint32_t nBitsTotal = 32;
    return nBitsTotal-nLeading0Bits-1/*isPow2(n)*/;
  }

  uint32_t ilog4_floor(const uint32_t n)
  {
    return (ilog2(n))/2;
  }

  uint32_t ilog4_ceil(const uint32_t n)
  {
    return ilog4_floor(n)+!isPow4(n)+(n==1);
  }

  uint32_t ipow2(const uint32_t exp)
  {
    return static_cast<uint32_t>(1) << exp;
  }

  uint32_t ipow4(const uint32_t exp)
  {
    return static_cast<uint32_t>(1) << (2*exp);
  }

  void test_imath()
  {
    for(size_t i=0; i<66; i++)
      std::cout << i << ": " << ilog2(i) << " " << ilog4_floor(i) << " " << ilog4_ceil(i) << " " << ipow4(ilog4_floor(i))<< " " << nextPow2(i) << std::endl;
  }
}

#endif //__HELPER_IMATH__
