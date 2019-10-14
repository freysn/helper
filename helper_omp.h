#ifndef __HELPER_OMP__
#define __HELPER_OMP__


namespace helper
{
  size_t getMaxNThreadsOMP()
  {
    return
#ifdef USE_OMP
      omp_get_max_threads();
#else
      1;
#endif    
  }

  void setNumThreadsOMP(int numThreads)
  {
#ifdef USE_OMP
    if(numThreads > 0)
      omp_set_num_threads(numThreads);
#endif
  }

  size_t getThreadId()
  {
    return 
    #ifdef USE_OMP
      omp_get_thread_num();
    #else
    0;
    #endif
  }
}


#endif //__HELPER_OMP__
