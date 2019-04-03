#ifndef __MLZSS_KERNEL_CU__
#define __MLZSS_KERNEL_CU__

#include "mlzss.h"

template<typename I>
__device__ __host__
I divUp(const I& a, const I& b)
{
  I o = a/b;
  o += (o*b) < a;
  return o;
}

#define MLZSS_NEW_THREAD_DISTRIBUTION

template<typename I>
__device__
I d_nElemsPerThread(const I& in_size, const I& nThreads)
{
#ifdef MLZSS_NEW_THREAD_DISTRIBUTION
  const I nElemsPerThread = divUp(in_size,nThreads);
#else
  const I nElemsPerThread = in_size/nThreads;
#endif
  return nElemsPerThread;
}

template<typename I>
__device__
I d_nElemsThisThread(const I& nElemsPerThread, const I& in_size, const I& threadId, const I& nThreads)
{
  I nElemsThisThread;
#ifdef MLZSS_NEW_THREAD_DISTRIBUTION
  nElemsThisThread =
    max(min((int)nElemsPerThread, (int)in_size-nElemsPerThread*threadId),0);
#else
  nElemsThisThread = nElemsPerThread;
  if(threadId == nThreads-1)
    nElemsThisThread = (in_size-nElemsPerThread*threadId);
#endif
  return nElemsThisThread;
}

#ifdef MLZSS_NEW_THREAD_DISTRIBUTION
#undef MLZSS_NEW_THREAD_DISTRIBUTION
#endif

template<typename I>
__device__
void d_lzss_thread_setup_dec(I& threadId, 			 
		       I& elemOffset, 
		       const I& in_size)
{
  threadId = threadIdx.x+blockIdx.x*blockDim.x;
  const I nThreads = gridDim.x*blockDim.x;
  const I nElemsPerThread = d_nElemsPerThread(in_size, nThreads);

  elemOffset = nElemsPerThread*threadId;  
}

//
// maybe split I and the out offset sometime in the future
//
template<typename I>
__device__
void d_lzss_thread_setup_enc(I& threadId, 
			 I& nElemsThisThread,
			 I& elemOffset,
			     I& nElemsPerThread,
			 const I& in_size)
{
  
  threadId = threadIdx.x+blockIdx.x*blockDim.x;
  const I nThreads = gridDim.x*blockDim.x;

  nElemsPerThread = d_nElemsPerThread(in_size, nThreads);  
  nElemsThisThread = d_nElemsThisThread(nElemsThisThread, in_size, threadId, nThreads);
  elemOffset = nElemsPerThread*threadId;
}


template<typename I>
__device__
void d_setThreadId(I& threadId)
{
  threadId = threadIdx.x+blockIdx.x*blockDim.x;
}

template<typename I>
__device__
void d_threadDataRange(I& nElemsPerThread, 
		       I& elemOffset,
		       I& nElemsThisThread,
		       const I& threadId,
		       const I& in_size)
{
  const I nThreads = gridDim.x*blockDim.x;
  nElemsPerThread = d_nElemsPerThread(in_size, nThreads);
  nElemsThisThread = d_nElemsThisThread(nElemsPerThread, in_size, threadId, nThreads);
  elemOffset = nElemsPerThread*threadId;
}

template<typename I>
__device__
void d_lzss_thread_setup_enc(I& threadId, 
			     I& nElemsThisThread,
			     I& elemOffset,			     
			     const I& in_size)
{  
  d_setThreadId(threadId);
  I nElemsPerThread;
  d_threadDataRange(nElemsPerThread,
		    elemOffset,
		    nElemsThisThread,
		    threadId,
		    in_size);
}

template<typename T, typename INFO, typename I,  typename ARGS, bool dryRun/*=false*/>
__global__
void g_lzss_encode(T* out_values,
		   INFO* out_info,
		   I* out_nElemsPerThread,
		   const T* in,
		   const I in_size,
		   const ARGS args)
{
  I threadId = 0;
  I elemOffset = 0;
  I nElemsThisThread = 0;
  d_lzss_thread_setup_enc(threadId, nElemsThisThread, 
		      elemOffset, in_size);

  const I size =
    lzss_encode<T, INFO, I, ARGS, dryRun>
  (out_values+elemOffset,
   out_info+elemOffset,
   in+elemOffset,
   nElemsThisThread,
   args);

  out_nElemsPerThread[threadId] = size;    
}


template<typename T,
  typename INFO,
  typename I,  
  typename ARGS>
  __global__
void g_lzss_decode(T* out,
		   const T* in_values,
		   const INFO* in_info,
		   const I* in_offsetsPerBlock,
		   const I* in_offsetsPerBlock_info,
		   const I out_size,                                                   
		   const ARGS args)
{
  I threadId = 0;
  I elemOffset = 0;
  
  d_lzss_thread_setup_dec(threadId, elemOffset, out_size);
  

  I in_elemOffset = 0;
  I in_elemOffset_info = 0;
  if(threadId > 0)
    {
      in_elemOffset = in_offsetsPerBlock[threadId-1];
      in_elemOffset_info = in_offsetsPerBlock_info[threadId-1];
    }
  const I nElems = in_offsetsPerBlock[threadId] - in_elemOffset;
  
  lzss_decode(out+elemOffset, in_values+in_elemOffset, in_info+in_elemOffset_info,
	      nElems, args);

  //lzss_decode(out+elemOffset, in_values+in_elemOffset, in_info+in_elemOffset,
  //in_nElemsPerThread[threadId], args);
}


template<unsigned int nBits, typename T, typename I>
__global__
void g_compress(T* out,
		const T* in,
		const I* offsets,
		const I out_size)
{
  I threadId = 0;
  I elemOffset = 0;
  
  d_lzss_thread_setup_dec(threadId, elemOffset, out_size);

  I offset = 0;
  if(threadId > 0)
    offset = offsets[threadId-1];

  const I offsetEnd = offsets[threadId];
  
  //
  // only required if nBits > 0
  //
  T bitTmp=0;
  T bitCnt=0;
  
  while(offset < offsetEnd)
    {
      const T v_in = in[elemOffset];
      if(nBits==0)
	{
	  out[offset] = v_in;
	  offset++;	  
	}
      else
	{
	  bitTmp |= (v_in << bitCnt);
	  bitCnt++;
	  bitCnt = (bitCnt % nBits);
	  if(bitCnt==0)
	    {
	      out[offset] = bitTmp;
	      bitTmp = 0;
	      offset++;
	    }
	}
      elemOffset++;
    }
  
  if(nBits > 0 && bitCnt > 0)
    out[offset] = bitTmp;
}

template<unsigned int nBits, typename T>
__global__
void g_iDivUp(T* out,
	      const T* in,
	      bool scannedOffsets)
{
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  T v = in[idx];
  if(scannedOffsets && idx > 0)
    v-=in[idx-1];
  
  out[idx] = v/nBits+(v%nBits > 0);
}

#endif //__MLZSS_KERNEL_CU__