#ifndef __LOAD_VOL__
#define __LOAD_VOL__

#include "UserData.h"
#include "volData/volData.h"
#include "VolDataHandlerCUDA.h"

#ifdef DATA_TEXTURE_BYTES

bool loadVol(ushort*& vol, int3& dim, char* configFileName, int& fileName, int& timeStep, int& nTimeSteps)
{
  UserData userData;
  userData.readConfig(configFileName);
  printf("after read config\n");
  dim.x = userData._volDim.x;
  dim.y = userData._volDim.y;
  dim.z = userData._volDim.z;
    
  while(fileName < 0)
    fileName += userData._volumeFiles.size();

  while(timeStep < 0)
    timeStep += userData._nTimeSteps;
  
  nTimeSteps = userData._nTimeSteps;
  
  fileName = fileName % userData._volumeFiles.size();
  timeStep = timeStep % userData._nTimeSteps;

  size_t size = dim.x*dim.y*dim.z;
  
  size_t offset = 0;

  if(userData._nTimeSteps > 1 && userData._sameFileTime == true)
    offset = timeStep*size*DATA_TEXTURE_BYTES;
  
  std::string volumeFileName = userData._volumeFiles[fileName];
  
  if(userData._nTimeSteps > 1 && userData._sameFileTime == false)
    {
      std::stringstream out;
      out << timeStep;
      std::string numStr = out.str();
      while(numStr.size() < userData._numFixedLen)
        numStr = "0" + numStr;
      volumeFileName += numStr + ".raw";
      std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ " << volumeFileName << std::endl;
    }

  std::cout << "offset: " << offset << std::endl;
#if DATA_TEXTURE_BYTES == 1
  assert(userData._volumeDataType == voldattype_uchar);
#endif
  
  if(userData._volumeDataType == voldattype_ushort)
    vol =  loadRawFileUShort((char*) volumeFileName.c_str(), size, offset);
  else if(userData._volumeDataType == voldattype_float)
    vol = loadRawFileFloat((char*)volumeFileName.c_str(), size, offset);
  else if(userData._volumeDataType == voldattype_uchar)
    {
#if DATA_TEXTURE_BYTES == 1
      vol = (ushort*) loadRawFileUCharPlain((char*)volumeFileName.c_str(), size, offset);
      //vol = loadRawFileUChar(volumeFileName.c_str(), size);
#else
      vol = loadRawFileUChar((char*)volumeFileName.c_str(), size, offset);
#endif
    }
  else
    {
      printf("data type not supported\n");
      exit(0);
    }
  return true;
}


bool updateVol(cudaArray*& dVol, uint3& volDim, std::string configFileName, int& timeStep, int& nTimeSteps)
{
  unsigned short* hVol = 0;
  
  //std::string configFileName = std::string("configs/engine.config");
  //std::string configFileName = std::string("configs/engine.config");

  int fileName = 0;
  //int timeStep = 0;
  
  int3 volDimI;
#if 0
  loadVol(hVol, volDimI, (char*)configFileName.c_str(), fileName, timeStep, nTimeSteps);
  volDim = make_uint3(volDimI.x, volDimI.y, volDimI.z);
#else
  VolDataHandlerCUDA<VolumeData_t> volDataHandler(configFileName);
  VolDataHandlerCUDA<VolumeData_t>::ret_t ret = volDataHandler.getH(timeStep, -1);
  volDimI = volDataHandler.getVolDim();
  volDim = make_uint3(volDimI.x, volDimI.y, volDimI.z);
  hVol = ret.data;
  timeStep = ret.timeStep;
  nTimeSteps = volDataHandler.getNTimeSteps();
  
#endif
  uploadVolume(dVol, hVol, volDimI);
  
  initVolumeTexs();
  bindVolTex(dVol);
    
#if 0
  delete [] hVol;
#endif
  return true;
}
#endif


#endif //__LOAD_VOL__
