#ifndef __HELPER_TEX_OBJ_CUDA__
#define __HELPER_TEX_OBJ_CUDA__

#include <iostream>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
namespace helper
{
  template<typename T>
  class helper_TexObjCUDA
  {
  public:
    helper_TexObjCUDA()
    {
      _buf = 0;
      _tex = 0;
    }

    template<typename I3>
    void init3D(T* data, I3 res)
    {
      const size_t N = res.x*res.y*res.z;
      cudaExtent extent = make_cudaExtent(res.x * sizeof(T),
					  res.y, res.z);
      
      if(_buf != 0)
	cudaFree(_buf);

      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
      checkCudaErrors(cudaMalloc3DArray(&_buf, &channelDesc, extent));

      // copy data to 3D array
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr   = make_cudaPitchedPtr(data, extent.width*sizeof(T), extent.width, extent.height);
      copyParams.dstArray = _buf;
      copyParams.extent   = extent;
      copyParams.kind     = cpyKind;
      checkCudaErrors(cudaMemcpy3D(&copyParams));

      
      // create texture object

      memset(&_resDesc, 0, sizeof(_resDesc));
      _resDesc.resType = cudaResourceTypeArray;
      _resDesc.res.array.array = _buf;
      
      //_resDesc.res.linear.sizeInBytes = N*sizeof(T);

      //_resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
      
      // if(std::is_same<float, T>::value)
      //   {
      // 	_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
      // 	_resDesc.res.linear.desc.x = 32; // bits per channel
      //   }

      // else
      //   {
      // 	 throw std::invalid_argument( "invalid type for cuda texture object" );
      //   }

    
      memset(&_texDesc, 0, sizeof(_texDesc));
      _texDesc.readMode =
	cudaReadModeElementType;
      //cudaReadModeNormalizedFloat;

      _texDesc.addressMode[0] = cudaAddressModeClamp;
      _texDesc.addressMode[1] = cudaAddressModeClamp;
      _texDesc.tex.addressMode[2] = cudaAddressModeClamp;

      _texDesc.filterMode = cudaFilterModePoint;

      cudaCreateTextureObject(&_tex, &_resDesc, &_texDesc, NULL);
    }

    cudaTextureObject_t get()
    {
      return _tex;
    }
  
  private:
    cudaResourceDesc _resDesc;
    cudaTextureDesc _texDesc;
    cudaPitchedPtr _buf;
    cudaTextureObject_t _tex;
  };
}

#endif //__HELPER_TEX_OBJ_CUDA__

