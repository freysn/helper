#ifndef __HELPER_CUDA_EVENT_TIMER__
#define __HELPER_CUDA_EVENT_TIMER__

/*
  RETURNS TIME IN SECONDS
 */
namespace helper
{
  template<bool startImmediately=false>
class CudaEventTimer
{
 public:
  CudaEventTimer()
    {
      cudaEventCreateWithFlags(&startEvent, _flag);
      cudaEventCreateWithFlags(&stopEvent, _flag);

      if(startImmediately)
	start();
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

  float get_s()
  {
    return stopGetTime();
  }
  
 private:
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  const unsigned int _flag = cudaEventDefault;

  //cudaEventBlockingSync;
  //cudaEventDefault
};
}

#endif //__HELPER_CUDA_EVENT_TIMER__
