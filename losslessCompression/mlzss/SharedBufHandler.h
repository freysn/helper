#ifndef __SHARED_BUF_HANDLER__
#define __SHARED_BUF_HANDLER__

#include <assert.h>
// assert() is only supported
// for devices of compute capability 2.0 and higher
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#if 1
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)) || defined(NDEBUG_CUDA)
#undef  assert
#define assert(arg)
#endif

//http://en.wikipedia.org/wiki/External_variable
// The extern keyword means "declare without defining"
#ifdef __CUDACC__
//extern __shared__ elem_t shared[];
__shared__ elem_t shared[nSharedMemElems];
#endif

template<typename T, typename I>
class SharedBufHandler
{
 public:
  __device__
    SharedBufHandler()
  {
    end_idx = 0;
    front_off = 0;
    buf_off = threadIdx.x*buf_size;
  }

  __device__
  T at(I idx)
  {
    assert(idx < end_idx);
    assert(idx >= end_idx-buf_size);
    const I off = idx-(end_idx-buf_size);
    assert(off >= 0);
    assert(off < buf_size);
    return shared[buf_off+(front_off+off) % buf_size];
  }

  /*
  __device__
  void load_until(const T* in, I new_end_idx)
  {
    if(new_end_idx>end_idx)
      load(in, new_end_idx-end_idx);
  }
  */
  
  __device__
    void load_from(const T* in, I new_from_idx, I max_end_idx)
  {
    const I from_idx = (end_idx-buf_size);
    if(new_from_idx > from_idx)
      load(in, min(new_from_idx-from_idx,
                   max_end_idx-end_idx));

    assert(end_idx <= max_end_idx);
  }
  
  __device__
  void load(const T* in, I nElems)
  {    
    for(I i=0; i<nElems; i++)
      {
        shared[buf_off+front_off] = in[end_idx];
        front_off = (front_off+1)%buf_size;
        end_idx++;
      }
  }

  I end_idx;
  I front_off;
  I buf_off;
};


template<typename T, typename I>
class SharedBufHandler_dummy
{
 public:
  __device__
    SharedBufHandler_dummy(const T* data)
  {
    this->data = data;
  }

  __device__
  T at(I idx)
  {
    return data[idx];
  }

  __device__
    void load(const T* /*in*/, I /*nElems*/)
  {    
  }
  
  const T* data;  
};


#endif //__SHARED_BUF_HANDLER__
