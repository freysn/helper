#ifndef __MLZSS_STD_ARGS_CPP__
#define __MLZSS_STD_ARGS_CPP__

#include "mlzss_std_args.h"
#include <cassert>

lzss_args_t<unsigned int> mlzss_init_args_uchar(unsigned int winSize)
{
  lzss_args_t<unsigned int> args;
  args.winSize = winSize;
  //args.winSize = 2;
  args.minLen = 2;
  args.encodeFac =16;
  assert(args.winSize < args.encodeFac);
  return args;
}

lzss_args_t<unsigned int> mlzss_init_args_ushort(unsigned int winSize)
{
  lzss_args_t<unsigned int> args;
  args.winSize = winSize;
  args.minLen = 2;
  args.encodeFac =256;
  assert(args.winSize < args.encodeFac);
  return args;
}

#endif //__MLZSS_STD_ARGS_CPP__
