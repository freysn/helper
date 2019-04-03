#ifndef __COMPRESSION_MLZSS__
#define __COMPRESSION_MLZSS__

#include "mlzss/mlzss.h"
#include "mlzss/mlzss_std_args.h"



unsigned int compress_mlzss(unsigned char* out_values,
                            unsigned char* out_info,
                            unsigned char* in,
                            unsigned int in_size)
{
  lzss_args_t<unsigned int> args = mlzss_init_args_uchar();

  unsigned int out_size =
    lzss_encode(&out_values[0], &out_info[0],
                &in[0], (unsigned int)in_size, args);
  return out_size;
}

unsigned int compress_mlzss_dry(unsigned char* in,
                                unsigned int in_size)
{
  lzss_args_t<unsigned int> args = mlzss_init_args_uchar();

  unsigned int out_size =
    lzss_encode<
    unsigned char,
    unsigned char,
    unsigned int,
    lzss_args_t<unsigned int>,
    true>
    (0, 0,
     &in[0], (unsigned int)in_size, args);
  return out_size;
}

unsigned int compress_mlzss_dry(unsigned short* in,
                                unsigned int in_size)
{
  lzss_args_t<unsigned int> args = mlzss_init_args_ushort();

  unsigned int out_size =
    lzss_encode<
    unsigned short,
    unsigned short,
    unsigned int,
    lzss_args_t<unsigned int>,
    true>
    (0, 0,
     &in[0], (unsigned int)in_size, args);
  return out_size;
}

#endif //__COMPRESSION_MLZSS__
