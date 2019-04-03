//#include "mlzss_params_auto.h"
#include "mlzss.cuh"
#include "mlzss_params.h"

#include "mlzss_kernel.cu"

#include "mlzss_std_args.cpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <ctime>


#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>



#include <helper_cuda.h>

#include <fstream>
#include <typeinfo>



template<typename T, typename I>
void mlzss_cuda_struct<T,I>::init(size_t nElems)
{

  this->nThreads = cuBlockDim.x*cuGridDim.x;
    
  this->nElems = nElems;
    
  cudaMalloc(&d_in, nElems*sizeof(T));
  cudaMalloc(&d_out_values, nElems*sizeof(T));
  cudaMalloc(&d_out_values_compressed, nElems*sizeof(T));
  // actually less is required here (for uchar, only 1/8th)
  cudaMalloc(&d_out_info, nElems*sizeof(T));
  cudaMalloc(&d_out_info_compressed, nElems*sizeof(T));

  //cudaMalloc(&d_out_nElemsPerThread, nThreads*sizeof(I));
  cudaMalloc(&d_out_offsetsPerBlock, nThreads*sizeof(I));
  cudaMalloc(&d_out_offsetsPerBlock_info, nThreads*sizeof(I));

  if(typeid(T) == typeid(unsigned char))
    args = mlzss_init_args_uchar(winSize);
  else if(typeid(T) == typeid(unsigned short))
    args =  mlzss_init_args_ushort(winSize);
  else
    assert(false);

  resetData();
}

template<typename T, typename I>
void mlzss_cuda_struct<T,I>::resetData()
{
  cudaMemset(d_in, 0, nElems*sizeof(T));
  cudaMemset(d_out_values, 0, nElems*sizeof(T));
  cudaMemset(d_out_values_compressed, 0, nElems*sizeof(T));
  // actually less is required here (for uchar, only 1/8th)
  cudaMemset(d_out_info, 0, nElems*sizeof(T));
  cudaMemset(d_out_info_compressed, 0, nElems*sizeof(T));

  ////cudaMemset(d_out_nElemsPerThread, 0, nThreads*sizeof(I));
  cudaMemset(d_out_offsetsPerBlock, 0, nThreads*sizeof(I));
  cudaMemset(d_out_offsetsPerBlock_info, 0, nThreads*sizeof(I));
}

template<typename T, typename I>
mlzss_cuda_struct<T,I>::~mlzss_cuda_struct()
{
  cudaFree(d_in);
  cudaFree(d_out_values);
  cudaFree(d_out_info);
  cudaFree(d_out_values_compressed);
  cudaFree(d_out_info_compressed);
  //cudaFree(d_out_nElemsPerThread);
  cudaFree(d_out_offsetsPerBlock);
  cudaFree(d_out_offsetsPerBlock_info);
}

template<typename T, typename I>
void mlzss_cuda_struct<T,I>::createOffsetsPerBlockInfo(bool scannedOffsets)
{
  g_iDivUp
    <8*sizeof(T)>
    <<<cuGridDim, cuBlockDim>>>
    (d_out_offsetsPerBlock_info,
     d_out_offsetsPerBlock, scannedOffsets);
    
  getLastCudaError("after iDivUp kernel");

  thrust::device_ptr<I> D_info(d_out_offsetsPerBlock_info);
  thrust::inclusive_scan(D_info, D_info+nThreads, D_info);    
}
  
template<typename T, typename I>
void mlzss_cuda_struct<T,I>::encode()
{
  g_lzss_encode
    <T,    
    T,
    unsigned int,
    lzss_args_t<unsigned int>,
    false>
    <<<cuGridDim, cuBlockDim
    //,buf_size*sizeof(elem_t)*cuBlockDim.x
    >>>
    (d_out_values, d_out_info, d_out_offsetsPerBlock,
     d_in, 
     (unsigned int)nElems, 
     args);

  getLastCudaError("after encode kernel");


  createOffsetsPerBlockInfo(false);
    
  thrust::device_ptr<I> D(d_out_offsetsPerBlock);
  thrust::inclusive_scan(D, D+nThreads, D);
        
  g_compress
    <0>
    <<<cuGridDim, cuBlockDim>>>
    (d_out_values_compressed, d_out_values,
     d_out_offsetsPerBlock, (unsigned int)nElems);

  g_compress
    <sizeof(T)*8>
    <<<cuGridDim, cuBlockDim>>>
    (d_out_info_compressed, d_out_info,
     d_out_offsetsPerBlock_info, 
     (unsigned int)nElems);
    

  getLastCudaError("after compression kernel");
}
  
template<typename T, typename I>
void mlzss_cuda_struct<T,I>::decode()
{
  g_lzss_decode
    <<<cuGridDim, cuBlockDim>>>
    (d_in, 
     d_out_values_compressed, 
     d_out_info_compressed,
     d_out_offsetsPerBlock, 
     d_out_offsetsPerBlock_info,
     (unsigned int)nElems, 
     args);
  getLastCudaError("after decode kernel");
}


// Use explicit instantiation to create the 
// versions of the template that will be used by the
// rest of the program.
//template mlzss_cuda_struct<elem_t, unsigned int>;
template class mlzss_cuda_struct <unsigned char, unsigned int>;

#ifdef MLZSS_WITH_MAIN

#include "files.cpp"

int main(int argc, char** argv)
{

  if(argc != 2)
    {
      std::cout << "one argument required: directory for input files\n";
      return 0;
    }
  
  //typedef unsigned char elem_t;

  
  
  
    
  srand(time(NULL));
  
  std::vector<elem_t> in, out_values, out_info, in_cmp;

  //
  // loop for testing
  //

  double totalTime = 0.;
  size_t totalBytes = 0;

  std::vector<std::vector<char> > in_data;
  {
    const bool success = dbg_read(in_data,
				  std::string(argv[1]));
    assert(success);
  }

  std::cout << "there are " << in_data.size() << " input data streams\n";
  for(size_t i=0; i<in_data.size(); i++)
    std::cout << in_data[i].size() << " bytes, ";
  std::cout << std::endl;
  
  mlzss_cuda_struct<elem_t, unsigned int> s;
  
  for(size_t i=0; i<in_data.size(); i++)
    {
      getLastCudaError("at the beginning of test iteration");
      {
	const size_t nBytes = in_data[i].size();
	in.resize(nBytes/sizeof(elem_t));
	std::memcpy(&in[0],
		    &in_data[i][0],
		    nBytes);

      }
      
      out_values.resize(in.size(),255);
      out_info.resize(in.size(),255);
      in_cmp.resize(in.size(),255);
  
      if(i==0)
	s.init(in.size());

      getLastCudaError("after test cuda struct init");
  
      cudaMemcpy(s.d_in, &in[0], in.size()*sizeof(elem_t),
		 cudaMemcpyHostToDevice);

      getLastCudaError("after test data upload");
  
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
  
      cudaEventRecord(start);

      getLastCudaError("after event generation");
      
      s.encode();
      /*
	g_lzss_encode
	<elem_t,    
	elem_t,
	unsigned int,
	lzss_args_t<unsigned int>,
	false>
	<<<cuGridDim, cuBlockDim
	//,buf_size*sizeof(elem_t)*cuBlockDim.x
	>>>
	(s.d_out_values, s.d_out_info, s.d_out_offsetsPerBlock,
	s.d_in, (unsigned int)in.size(), args);

	getLastCudaError("after encode kernel");
      */
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);      
  
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalTime += milliseconds/1000.;
      std::cout << "the operation took " << milliseconds << " ms\n";
      
      getLastCudaError("after elapsed encode time");

      {
	/*
	thrust::device_ptr<unsigned int>
	  D(s.d_out_offsetsPerBlock);
	const size_t nElems =
	  thrust::reduce(D, D+s.nThreads,
			 (int) 0, thrust::plus<int>());
	*/
	unsigned int nElems;
	cudaMemcpy(&nElems,
		   &s.d_out_offsetsPerBlock[s.nThreads-1],
		   sizeof(unsigned int),
		   cudaMemcpyDeviceToHost
		   );
	totalBytes += nElems*sizeof(elem_t);
	
	std::cout << "resulting size: " 
		  <<  nElems*sizeof(elem_t)
		  << " orig: "<< in.size()
		  << std::endl;
      }

      cudaMemset(s.d_in, 255, in.size()*sizeof(elem_t));

      {
	const size_t off = 27;
	std::vector<unsigned char> tmp;
	size_t nBytes = s.serialize_toHost(tmp, off);
	std::cout << "size: " << nBytes << std::endl;
	s.resetData();
	s.deserialize_fromHost(&tmp[off]);
      }
      
      s.decode();
      /*
      g_lzss_decode
	<<<cuGridDim, cuBlockDim>>>
	(s.d_in, 
	 s.d_out_values, 
	 s.d_out_info,
	 s.d_out_offsetsPerBlock, 
	 (unsigned int)in.size(), 
	 args);
      */

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
  
      cudaMemcpy(&in_cmp[0], s.d_in,
		 in.size()*sizeof(elem_t),
		 cudaMemcpyDeviceToHost);


      for(size_t i=0; i<in.size(); i++)
	{
	  if(in_cmp[i] == in[i])
	    continue;
	  std::cout << i << " "
		    << (int) in[i]
		    << " " << (int) in_cmp[i]
		    << std::endl;
	  assert(false);
	}      
    }

  std::ofstream outfile;

  outfile.open(log_fname.c_str(), std::ios_base::app);

  outfile << sizeof(elem_t)
	  << ", " << winSize
	  << ", " << buf_size
	  << ", " << joint_warp_buf_update
	  << ", " << cuBlockDim.x
	  << ", " << cuGridDim.x
	  << ", ";
    outfile << totalTime/in_data.size()
	  << ", " << totalBytes/in_data.size();
  outfile << std::endl;
  
  std::cout << "On average, encoding took " << totalTime/in_data.size() << " s"
	    << ", resulting in file size " << totalBytes/in_data.size() << " bytes"
	    << std::endl;
  
  
  return 0;
}

#endif