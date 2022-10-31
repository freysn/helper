#ifndef __MLZSS_CUH__
#define __MLZSS_CUH__

#include "mlzss_params.h"
#include "mlzss_std_args.h"
#include <cassert>
#include <helper_cuda.h>
#include <iostream>

template<typename T, typename I>
class mlzss_cuda_struct
{
public:

  void init(size_t nElems);

  void resetData();
  
  ~mlzss_cuda_struct();

  void createOffsetsPerBlockInfo(bool scannedOffsets);
  
  void encode();
  
  template<typename U>
  void deviceCopy_encode(U* d_in)
  {
    cudaMemcpy(this->d_in, d_in, nElems*sizeof(T), cudaMemcpyDeviceToDevice);
    encode();
  }
  
  void decode();

  template<typename U>
  size_t serialize_toHost(U& msg, size_t off)
  {    
    size_t nBytes = nThreads*sizeof(I);
    msg.resize(off+nBytes);
    cudaMemcpy(&msg[off],
	       d_out_offsetsPerBlock,
	       nBytes,
	       cudaMemcpyDeviceToHost);
    
    const I nElems_enc = ((I*)(&msg[off]))[nThreads-1];
    off += nBytes;
    
    I nElems_info = 0;
    cudaMemcpy(&nElems_info,
	       &d_out_offsetsPerBlock_info[nThreads-1],
	       sizeof(I),
	       cudaMemcpyDeviceToHost);

    //std::cout << "aaaaaaaaaaaaaaaa " 
    //<< nElems_enc << " " << nElems_info << std::endl;

    msg.resize(/*nThreads*sizeof(I)*/off+sizeof(T)*(nElems_enc+nElems_info));

    cudaMemcpy(&msg[off],
	       d_out_values_compressed,
	       nElems_enc*sizeof(T),
	       cudaMemcpyDeviceToHost);

    off+= nElems_enc*sizeof(T);
    cudaMemcpy(&msg[off],
	       d_out_info_compressed,
	       nElems_info*sizeof(T),
	       cudaMemcpyDeviceToHost);

    off += nElems_info*sizeof(T);
    //std::cout << "aaaaaaaaaaa " << nElems <<" " <<nElems_info <<std::endl;
    return off;
  }

  template<typename U>
  void deserialize_fromHost(U msg)
  {
    size_t off=0;
    size_t nBytes = nThreads*sizeof(I);
    cudaMemcpy(d_out_offsetsPerBlock,
	       &msg[off],
	       nBytes,
	       cudaMemcpyHostToDevice);
    
    const I nElems_enc = ((I*)(&msg[off]))[nThreads-1];
    
    off += nBytes;
    
    createOffsetsPerBlockInfo(true);

    I nElems_info = 0;
    cudaMemcpy(&nElems_info,
	       &d_out_offsetsPerBlock_info[nThreads-1],
	       sizeof(I),
	       cudaMemcpyDeviceToHost);

    //std::cout << "aaaaaaaaaaaaaaaa " 
    //<< nElems_enc << " " << nElems_info << std::endl;

    nBytes = nElems_enc*sizeof(T);
    cudaMemcpy(d_out_values_compressed,
	       &msg[off],
	       nBytes,
	       cudaMemcpyHostToDevice);
    off += nBytes;

    nBytes = nElems_info*sizeof(T);
      
    cudaMemcpy(d_out_info_compressed,
	       &msg[off],	       
	       nBytes,
	       cudaMemcpyHostToDevice);

    off += nBytes;

    //std::cout << "aaaaaaaaaaa " << nElems <<" " <<nElems_info <<std::endl;
    //assert(msg.size()==off);

    getLastCudaError("after deserialization");
  }

  size_t getNBytesInput()
  {
    return sizeof(T)*nElems;
  }
  
  T* d_in;
  T* d_out_values;
  T* d_out_info;

  T* d_out_values_compressed;
  T* d_out_info_compressed;

  //I* d_out_nElemsPerThread;
  I* d_out_offsetsPerBlock;
  I* d_out_offsetsPerBlock_info;

  size_t nElems;
  size_t nThreads;
  lzss_args_t<unsigned int> args;
};

#endif //__MLZSS_CUH__