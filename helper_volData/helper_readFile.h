#ifndef __HELPER_READ_FILE__
#define __HELPER_READ_FILE__

#ifndef NO_CIMG
#include "helper/helper_cimg.h"
#endif
//#include "read_FS3D_impl.h"
//#include "fs3dreader_impl.h"
#include "UserData.h"
#include <vector>
#include <string>
#include <limits>

// #ifndef NO_BZIP2
// #include "losslessCompression/bzip_helper/bzip_helper.h"
// #endif

namespace helper
{
#if 0
  template<class T> 
    T reverseFloatEndianness(const T t)
    {
      return t;
    }

  template<>
  float reverseFloatEndianness<float>( const float inFloat)
{
   float retVal;
   char *floatToConvert = ( char* ) & inFloat;
   char *returnFloat = ( char* ) & retVal;

   // swap the bytes into a temporary buffer
   returnFloat[0] = floatToConvert[3];
   returnFloat[1] = floatToConvert[2];
   returnFloat[2] = floatToConvert[1];
   returnFloat[3] = floatToConvert[0];

   return retVal;
}

  template<>
  float3 reverseFloatEndianness<float3>( const float3 inFloat3)
  {
    float3 retVal;
    
    retVal.x = reverseFloatEndianness(inFloat3.x);
    retVal.y = reverseFloatEndianness(inFloat3.y);
    retVal.z = reverseFloatEndianness(inFloat3.z);
    return retVal;
  }
#endif

  template<typename V>
int64_t readRawFile(std::vector<V>& buf, std::string volumeFileName, size_t offsetElems=0, size_t nElems=std::numeric_limits<size_t>::max())
{
#ifdef VERBOSE
  std::cout << "Open File: " << volumeFileName << std::endl;
#endif
  std::ifstream file(volumeFileName.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

  if(!file.is_open())
    {
      std::cout << "could not open file: " << volumeFileName << std::endl;
      //assert(file.is_open());
      return -1;
    }
  
  size_t size = file.tellg();

  if(nElems == std::numeric_limits<size_t>::max())
    nElems = size/sizeof(V) - offsetElems;

  if(!((offsetElems+nElems)*sizeof(V) <= size))
    std::cerr << __func__ << " expected " << (offsetElems+nElems)*sizeof(V) << " vs data size " << size << std::endl;
  assert((offsetElems+nElems)*sizeof(V) <= size);
  
  
  file.seekg (sizeof(V)*offsetElems, std::ios::beg);
  buf.resize(nElems);
  //assert(buf.size() * sizeof(V) == size);
  file.read((char*)&buf[0], nElems*sizeof(V));
      
  return buf.size();
}

//   template<typename V, typename UserData_T>
//     size_t readFile_FS3D(std::vector<V>& buf, std::string volumeFileName,
// 			 const UserData_T& userData,
// 			 size_t offsetElems=0,
// 			 size_t nElems=std::numeric_limits<size_t>::max())
//     {
//       	  std::string fname = volumeFileName;
// #if 1
// 	  const bool readDomain = true;
// 	  int res[3];
// 	  int numcomp;
// 	  std::vector<double> timestepvalues;
// 	  readDataInfo(fname, readDomain, res, 
// 		       numcomp, timestepvalues);
// 	  //assert(numcomp == 1);

// 	  //std::vector<float> data;
	  
// 	  buf.resize((res[0]*res[1]*res[2]*numcomp*sizeof(float))/sizeof(V));

// 	  int subext[6];
// 	  for(int d=0; d<3; d++)
// 	    {
// 	      subext[2*d] = 0;
// 	      subext[2*d+1] = res[d]-1;
// 	    }
// 	  readData(fname, res, subext,
// 		   numcomp, (float*) &buf[0]);
// 	  return buf.size();
// #else
// 	  const int isBinary = checkIfBinary(fname.c_str());

// 	  std::cout << fname << " is binary: " << isBinary << std::endl;

// 	  int endianness;
// 	  int resolution[3];

// 	  int numberOfComponents;
// 	  int numberOfDimensions;
// 	  float time;
  
// 	  getDataInfo(fname.c_str(), isBinary, 
// 		      endianness, resolution, 
// 		      numberOfComponents, numberOfDimensions, 
// 		      time);

// 	  std::cout << fname
// 		    << " is binary: " << isBinary
// 		    << ", endianness " << endianness
// 		    << ", resolution " << resolution[0]
// 		    << "x" << resolution[1] << "x" << resolution[2]
// 		    << ", #componens " << numberOfComponents
// 		    << ", #dimensions " << numberOfDimensions
// 		    << ", time " << time
// 		    << std::endl;

  
// 	  float* data = readFS3DBinary(fname.c_str(), resolution, 
// 				       numberOfComponents, endianness);


// 	  const size_t n = resolution[0]*resolution[1]*resolution[2];
	  
// 	  buf.resize((sizeof(float)/sizeof(V))*n);
// 	  memcpy(&buf[0], data, sizeof(float)*n);
// 	  free(data);
// 	  return buf.size();
// #endif
//     }
  
  template<typename V, typename UserData_T>
    size_t readFile(std::vector<V>& buf, std::string volumeFileName,
		    const UserData_T& userData,
		    size_t offsetElems=0, size_t nElems=std::numeric_limits<size_t>::max())
    {
      if(userData._volumeFormat == volformat_raw)
	{
	  const size_t result =
	    readRawFile(buf, volumeFileName, offsetElems, nElems);
	  /*
	  if(false)
	    for(size_t i=0; i<buf.size(); i++)
	      buf[i] = reverseFloatEndianness(buf[i]);
	  */
	  return result;
	  
	}
#ifndef NO_BZIP2
      else if(userData._volumeFormat == volformat_raw_bz2)
	{
          bzip_decompress(buf, volumeFileName);                    
	  return buf.size();
	}
#endif
      else if(userData._volumeFormat == volformat_png)
	{
#ifndef NO_CIMG

	  struct
	  {
	    int x;
	    int y;
	    int z;
	  }
	    volDim;
	  int nChannels;
	  //std::cout << "read image file: " << volumeFileName << std::endl;
	  cimgRead(buf, volDim, nChannels, volumeFileName, true);

	  assert(volDim.x == userData._volDim.x);
	  assert(volDim.y == userData._volDim.y);
	  assert(volDim.z == userData._volDim.z);
	  assert(nChannels == 1);
	  return buf.size();
#else
          assert(false);
          return 0;
#endif
	}
      // else if(userData._volumeFormat == volformat_fs3d)
      // 	{
      // 	  return readFile_FS3D(buf, volumeFileName,
      // 			       userData,
      // 			       offsetElems,
      // 			       nElems);
      // 	}
      assert(false);
      return 0;
    }
};

#endif //__HELPER_READ_FILE__
