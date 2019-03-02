#ifndef __HELPER_MULTIGPU_CUDA__
#define __HELPER_MULTIGPU_CUDA__

namespace helper
{
  int setDeviceCUDA_latestComputeMode(int forceDevice=-1)
  {
    int selectedDev = -1;
  
    int nDevs;
    checkCudaErrors(cudaGetDeviceCount(&nDevs));

    std::pair<int, int> latestComputeMode(0,0);
  
    for(int i = 0; i < nDevs; ++i)
      {
	const int kb = 1024;
	const int mb = kb * kb;
      
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

	wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;

	std::pair<int, int> computeMode(props.major, props.minor);
	
	if(std::get<0>(computeMode) > std::get<0>(latestComputeMode)
	   || ((std::get<0>(computeMode) == std::get<0>(latestComputeMode)) &&
	       (std::get<1>(computeMode) > std::get<1>(latestComputeMode))))
	  {
	    latestComputeMode = computeMode;
	    selectedDev = i;
	  }	
      }

    wcout << " select device " << selectedDev << std::endl;
    assert(selectedDev != -1);

    if(forceDevice >= 0)
      {
	selectedDev = forceDevice;
	wcout << "force selection of device " << selectedDev << std::endl;
      }
    
    checkCudaErrors(cudaSetDevice(selectedDev));

    return selectedDev;
  }

  size_t getDeviceCUDA_globalMem(int i)
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    return props.totalGlobalMem;
  }
};

#endif //__HELPER_MULTIGPU_CUDA__
