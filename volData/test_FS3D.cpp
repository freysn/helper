#include "read_FS3D_impl.h"
#include <iostream>

int main(int argc, char** argv)
{
  std::string fname = argv[1];
  const int isBinary = checkIfBinary(fname.c_str());

  std::cout << fname << " is binary: " << isBinary << std::endl;

  int endianness;
  int resolution[3];

  int numberOfComponents;
  int numberOfDimensions;
  float time;
  
  getDataInfo(fname.c_str(), isBinary, 
	      endianness, resolution, 
	      numberOfComponents, numberOfDimensions, 
	      time);

  std::cout << fname
	    << " is binary: " << isBinary
	    << ", endianness " << endianness
	    << ", resolution " << resolution[0]
	    << "x" << resolution[1] << "x" << resolution[2]
	    << ", #componens " << numberOfComponents
	    << ", #dimensions " << numberOfDimensions
	    << ", time " << time
	    << std::endl;

  
  float* data = readFS3DBinary(fname.c_str(), resolution, 
			       numberOfComponents, endianness);
  
  return 0;
}
