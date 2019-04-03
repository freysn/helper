// generated automatically by mlzss_tune.py
#ifndef __MLZSS_PARAMS__
#define __MLZSS_PARAMS__
#include <string>
#include "mlzss_args.h"
#include "mlzss_std_args.h"
const unsigned int winSize = 15;
const unsigned int buf_size = 32;
const bool joint_warp_buf_update = 1;
const dim3 cuBlockDim(256);
const dim3 cuGridDim(32);
const std::string log_fname("log.csv");
typedef unsigned char elem_t;
const unsigned int nSharedMemElems = 8192;
#endif
