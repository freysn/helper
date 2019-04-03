#include <iostream>
#include "VolDataHandlerCUDA.h"

typedef unsigned char volType_t;
int main(int argc, char** argv)
{
  if(argc!=2)
    {
      std::cout << "exactly one config file required as argument\n";
      exit(0);
    }
  
  VolDataHandlerCUDA<volType_t> volDataHandler
    = VolDataHandlerCUDA<volType_t>(std::string(argv[1]));
  
  const int3 volDim = volDataHandler.getVolDim();
  std::cout << "The volume has the following dimensions: " 
            << volDim.x << " "
            << volDim.y << " "
            << volDim.z << std::endl;

  const int nTimeSteps = volDataHandler.getNTimeSteps();
  std::cout << "The data set has " << nTimeSteps << " time steps\n";
    
  // load volume data into host memory
  volDataHandler.toHostMemory();

  // load the volumes at a random time step
  int timeStep = 0;
  volType_t* data = volDataHandler.getH(timeStep, -1).data;

  // do something with the data ...
  
  return 0;
}
