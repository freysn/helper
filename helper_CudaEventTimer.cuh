#ifndef __HELPER_CUDA_EVENT_TIMER__
#define __HELPER_CUDA_EVENT_TIMER__

/*
  RETURNS TIME IN SECONDS
 */
namespace helper
{
class CudaEventTimer
{
 public:
  CudaEventTimer()
    {
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);
    }

  ~CudaEventTimer()
    {
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
    }
  
  void start()
  {
    cudaEventRecord(startEvent, 0);
  }
  
  float stopGetTime()
  {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    
    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    
    //convert milliseconds to seconds
    return time*0.001f;
  }
  
 private:
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
};
}

#endif //__HELPER_CUDA_EVENT_TIMER__
