#ifndef __MLZSS_EST_QUANT_CUH__
#define __MLZSS_EST_QUANT_CUH__

#include "mlzss_params.h"

void mlzss_est_quant(std::vector<int>& est_quant,
		     unsigned int* d_out_nElemsPerThread,
		     const elem_t* d_in,
		     const size_t nBytesPerConfig,
		     const size_t nConfigs);

#endif //__MLZSS_EST_QUANT_CUH__