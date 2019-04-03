//
// estimate the impact of different quantization factors on the resulting file size
//
#include "mlzss_params.h"

#include "mlzss_kernel.cu"
#include "mlzss_std_args.h"

#include <vector>

#include <helper_cuda.h>
#include <typeinfo>
//#include <cassert>

#include <thrust/scan.h>
#include <thrust/device_vector.h>

template<typename I, typename ARGS>
__global__
void g_lzss_est_quant(I* out_nElemsPerThread,
		      const elem_t* in,
		      const I nElemsPerConfig,
		      const I nConfigs,
		      const ARGS args)
{
  I threadId;
  d_setThreadId(threadId);

  I elemOffset = 0;
  I nElemsThisThread = 0;
  I data_id = (threadId/nConfigs)*nConfigs;
  I configId = threadId - data_id;

  assert(configId < nConfigs);
  
  I nElemsPerThread = 0;
  d_threadDataRange(nElemsPerThread,
		    elemOffset,
		    nElemsThisThread,		    
		    data_id,
		    nElemsPerConfig);
  
  const bool dryRun = true;
  const I size =
    lzss_encode<elem_t, /*void*/char, I, ARGS, dryRun>
    (0, 0,
     &in[nElemsPerConfig*configId+data_id*nElemsPerThread],
     nElemsThisThread,
     args);


  const I nThreads = blockDim.x*gridDim.x;
  const I nThreadsPerQuant = (nThreads/nConfigs);
  assert(configId < nThreadsPerQuant);

  const I id = threadId/nConfigs
    +configId*nThreadsPerQuant;
  assert(id < nThreads);

  out_nElemsPerThread[id] = size;
  //out_nElemsPerThread[id] = configId;

  //out_nElemsPerThread[threadId] = 1;

}


//template<typename F>
void mlzss_est_quant(std::vector<int>& est_quant,
		     unsigned int* d_out_nElemsPerThread,
		     const elem_t* d_in,
		     const size_t nBytesPerConfig,
		     const size_t nConfigs)
{
  const size_t nElemsPerConfig = nBytesPerConfig/sizeof(elem_t);
  
  typedef unsigned int I;
  //typedef float F;

  lzss_args_t<unsigned int> args;
  if(typeid(elem_t) == typeid(unsigned char))
    args = mlzss_init_args_uchar(winSize);
  else if(typeid(elem_t) == typeid(unsigned short))
    args =  mlzss_init_args_ushort(winSize);
  else
    assert(false);
  
  const I nThreads = cuBlockDim.x*cuGridDim.x;  
  
  

  /*
  float* d_quantizationFactors = 0;
  cudaMalloc(&d_quantizationFactors, quantizationFactors.size()*sizeof(float));
  cudaMemcpy(d_quantizationFactors,
	     &quantizationFactors[0],
	     quantizationFactors.size()*sizeof(float),
	     cudaMemcpyHostToDevice);
  */


  const bool time_kernel = false;
  
  cudaEvent_t start, stop;
  if(time_kernel)
    {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
  
      cudaEventRecord(start);
    }
  
  g_lzss_est_quant<<<cuGridDim, cuBlockDim>>>(d_out_nElemsPerThread,
					      d_in,
					      (unsigned int)nElemsPerConfig,
					      (unsigned int)nConfigs,
					      args);
  
  getLastCudaError("after g_lzss_est_quant");

  if(time_kernel)
    {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);      
  
  
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      std::cout << "the operation took " << milliseconds << " ms\n";

      getLastCudaError("after elapsed encode time");
    }

  thrust::device_ptr<I> D(d_out_nElemsPerThread);
  thrust::inclusive_scan(D, D+nThreads, D);

  std::vector<I> out_nElemsPerThread(nThreads);
  cudaMemcpy(&out_nElemsPerThread[0],
	     d_out_nElemsPerThread,
	     nThreads*sizeof(I),
	     cudaMemcpyDeviceToHost);

  //for(size_t i=0; i<out_nElemsPerThread.size(); i++)
  //std::cout << out_nElemsPerThread[i] << " ";
  //std::cout << std::endl;

  assert(nThreads % nConfigs == 0);
  
  est_quant.resize(nConfigs);
  for(size_t i=0; i<est_quant.size(); i++)
    est_quant[i] =
      out_nElemsPerThread
      [(i+1)*(nThreads/nConfigs)-1];

  for(size_t i=est_quant.size()-1; i>0; i--)
    est_quant[i] -= est_quant[i-1];

  for(size_t i=0; i<est_quant.size(); i++)
    est_quant[i] *= nConfigs;

  for(size_t i=0; i<est_quant.size(); i++)
    est_quant[i] +=
      nThreads*sizeof(I) + (est_quant[i]/(sizeof(elem_t)*8));
  
  getLastCudaError("after g_lzss_est_quant");
}


#ifdef MLZSS_EST_QUANT_MAIN

#include "mlzss_std_args.cpp"
#include "mlzss.cuh"
#include "mlzss.cu"

int main(int argc, char** argv)
{
  srand(/*28*/time(NULL));
  std::vector<float> in(1024*1024, 0.f);

  for(size_t i=0; i<in.size(); i++)
    {
      in[i] = ((float)rand()/RAND_MAX)*128.f;
      //std::cout << in[i] << " " << std::endl;
    }

  const unsigned int nConfigs = 8;
  
  unsigned short* d_in;
  cudaMalloc(&d_in,
	     in.size()*sizeof(unsigned short)
	     *nConfigs);
  
  

  for(size_t i=0; i<nConfigs; i++)
    {
      const float quantFactor = (i+1)*2;

      std::vector<unsigned short> v(in.size());
      for(size_t j=0; j<v.size(); j++)
	v[j] = in[j]/quantFactor;

      cudaMemcpy(&d_in[i*v.size()], &v[0],
		 v.size()*sizeof(unsigned short),
		 cudaMemcpyHostToDevice);
    }


  unsigned int* d_out_nElemsPerThread=0;
  cudaMalloc(&d_out_nElemsPerThread,
	     cuBlockDim.x*cuGridDim.x*sizeof(unsigned int));
    
  std::vector<int> est_quant;
  mlzss_est_quant(est_quant, 
		  d_out_nElemsPerThread,
		  (unsigned char*)d_in,
		  in.size()*sizeof(unsigned short),
		  nConfigs);

  std::cout << "raw estimation sizes: \n";
  for(size_t i=0; i<est_quant.size(); i++)
    std::cout << i << ": " << est_quant[i] << std::endl;

  {
    mlzss_cuda_struct<unsigned char, unsigned int> mlzss;
    mlzss.init(in.size()*sizeof(unsigned short));

    for(size_t i=0; i<nConfigs; i++)
      {
	mlzss.deviceCopy_encode(&d_in[i*in.size()]);
	std::vector<unsigned char> data;
	mlzss.serialize_toHost(data, 0);
	std::cout << "full size: " << data.size() << std::endl;
      }
  }
  
  cudaFree(d_in);
  return 0;
}
#endif