#ifndef __MLZSS_ARGS__
#define __MLZSS_ARGS__

template<typename I>
class lzss_args_t
{
 public:
  I winSize;
  I minLen;
  I encodeFac;
};

#endif //__MLZSS_ARGS__
