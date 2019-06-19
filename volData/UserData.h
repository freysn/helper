#ifndef __USERCONFIG__
#define __USERCONFIG__

#include <typeinfo>

#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <tuple>


//#include "UserData.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>
//#include "read_FS3D_impl.h"
#include "fs3dreader_impl.h"

#define M_VEC
#include "vec.h"


//#include <vector_types.h>

/*
  Config File Syntax:
  -------------------
  Lines starting with # are ignored
  General Form:
  KEYWORD ARG_1 ARG_2 ... ARG_N
  
  Valid Keywords:
  
  LEAP [BOOL]
  RENDER [BOOL]
  VOLUME_FILE [STRING]
  REGION_FILE_LOAD [STRING]
  REGION_FILE_SAVE [STRING]
  VOLUME_DIM [INT] [INT] [INT]
  VOLUME_DATA_TYPE
 */

enum voldattype {
  voldattype_ushort, 
  voldattype_float,
  voldattype_float2,
  voldattype_float3,
  voldattype_uchar, 
  voldattype_double,
  voldattype_double3,
  voldattype_uchar4, 
  voldattype_none,  
};

enum volformat {
  volformat_raw,
  volformat_raw_bz2,
  volformat_png,
  volformat_fs3d=896,  
  volformat_none,
};

enum voldattoscalar {voldattoscalar_x, voldattoscalar_y, voldattoscalar_z, voldattoscalar_mag, voldattoscalar_none};

template<typename int3=V3<int>, typename float3=V3<float>, typename uchar4=V4<unsigned char>>
class UserData
{
  /*
  typedef V3<int> int3;
  typedef V3<float> float3;
  typedef V4<unsigned char> uchar4;
  */
  
  //
  //using namespace std;
  const char* letterString = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
  const char* numberString = "01234567890.-";
  const char* pathString = "/._-:\\~";
  
  /*
  struct int3
  {
    int x;
    int y;
    int z;
  };
*/
  
  
 public:

/*
  struct float3
  {
    float x;
    float y;
    float z;
  };

  struct uchar4
  {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
  };
*/
  
  UserData()
    {
      init();
    }

  size_t getTypeHash() const
  {
    return std::get<0>(getType());
  }
  
  std::tuple<size_t,size_t, size_t, size_t> getType() const
  {
    //#ifndef NO_CPP11
#if __cplusplus >= 201103L
    switch(_volumeDataType)
      {
      case voldattype_ushort:
        return std::make_tuple
	  (typeid(unsigned short).hash_code(),
	   typeid(unsigned short).hash_code(),
	   sizeof(unsigned short),
	   1);
      case voldattype_float: 
        return std::make_tuple
	  (typeid(float).hash_code(),
	   typeid(float).hash_code(),
	   sizeof(float),
	   1)
	   ;
      case voldattype_float3:
        return std::make_tuple
	  (typeid(float3).hash_code(),
	   typeid(float).hash_code(),
	   sizeof(float),
	   3)
	   ;
	case voldattype_float2:
        return std::make_tuple
	  (typeid(float2).hash_code(),
	   typeid(float).hash_code(),
	   sizeof(float),
	   2)
	   ;
      case voldattype_uchar: 
        return std::make_tuple
	  (typeid(uint8_t).hash_code(),
	   typeid(uint8_t).hash_code(),
	   sizeof(uint8_t),
	   1);
      case voldattype_uchar4: 
        return std::make_tuple
	  (typeid(uchar4).hash_code(),
	   typeid(unsigned char).hash_code(),
	   sizeof(unsigned char),
	   4);
      case voldattype_double:
        return std::make_tuple
	  (typeid(double).hash_code(),
	   typeid(double).hash_code(),
	   sizeof(double),
	   1)
	   ;
        //case voldattype_double3:
        //return typeid(double3).hash_code();
      case voldattype_none:
      default:
        break;
      };
    assert(false);
#else
    #error "require c++11"
#endif
    return std::make_tuple(0, 0,0,0);
  }

  
  

  static void getArguments(std::string line, std::vector<std::string>& args)
  {    
    const char* argString = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890.-/._-:\\$*~";
 
    int index, index2;
    //get string arguments
    args.clear();
  
    index = line.find_first_of(argString);
    //just make it not equal to string::npos for now
    index2 = (int)std::string::npos+3;
    while(index != (int)std::string::npos && index2 != (int)std::string::npos)
      {
        index2 = line.find_first_not_of(argString, index);
        args.push_back(line.substr(index, index2-index));
        if(index2 != (int)std::string::npos)
          {
            line = line.substr(index2);
            index = line.find_first_of(argString);
          }
      }
  }

  
  static std::vector<std::string> call_cmd(std::string command)
  {
    FILE *fpipe;
  char line[1024];
  
  if(!(fpipe = (FILE*)popen(command.c_str(),"r")))
    {  // If fpipe is NULL
      perror("Problems with pipe");
      //exit(1);
      assert(false);
    }
  
  std::vector<std::string> out;
   while ( fgets( line, sizeof line, fpipe))
   {
     std::string str(line);
     
     str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
     out.push_back(str);
     //printf("myprogram: %s", line);
   }
   pclose(fpipe);
   return out;
  }

  bool readConfig(const std::string& configFileName)
  {
    return readConfig(configFileName.c_str());
  }

  //bool readConfig(const char* configFileName);

  //bool readConfigOnly(const char* configFileName);


  bool writeConfig(const std::string& configFileName)
  {
    std::ofstream outfile(configFileName);
    if(!outfile.is_open())
      return false;

    assert(_volumeFiles.size()==1);
    outfile << "VOLUME_FILE " << _volumeFiles[0] << std::endl;
    outfile << "VOLUME_FORMAT ";

    switch(_volumeFormat)
      {
      case volformat_raw:
        outfile << "RAW";
        break;
      case volformat_raw_bz2:
        outfile << "RAW_BZ2";
        break;
      case volformat_png:
        outfile << "PNG";
        break;
      default:
          std::cout << __func__ << " ERROR, UNKNOWN FORMAT " << _volumeFormat << std::endl;
      };
    
    outfile << std::endl;
    outfile << "VOLUME_DIM " << _volDim.x << " " << _volDim.y << " " << _volDim.z << std::endl;
    outfile << "VOLUME_DATA_TYPE ";
    switch(_volumeDataType)
      {
      case voldattype_float:
	outfile << "FLOAT";
	break;
      case voldattype_uchar:
	outfile << "UCHAR";
	break;
      default:
	std::cout << __PRETTY_FUNCTION__ << " | data type not supported\n";
	outfile << "UNKNOWN";
      }
    outfile << std::endl;
    outfile << "VOXEL_SIZE " << _voxelSize.x << " " << _voxelSize.y << " " << _voxelSize.z << std::endl;
    return true;
  }

  std::vector<std::string> getFileNameBuf() const
    {
      return _fileNameBuf;
    }

  std::string getFileName(size_t t) const
    {
      #ifndef NDEBUG
      if(t>=_fileNameBuf.size())
	{
	  std::cout << "request index " << t << " of fileNameBuf of size " << _fileNameBuf.size() << ":";
	  for(auto e : _fileNameBuf)
	    std::cout << " " << e;
	  std::cout << std::endl;
	}
      #endif
      assert(t<_fileNameBuf.size());
      return _fileNameBuf[t];
    }
  
  std::vector<std::string> genFileNames()
    {      
      if(_volumeFiles.size() > 1)
        {
          std::cout << "found more than one volume file: ";
          for(auto v : _volumeFiles)
            std::cout << " " << v;
          std::cout << std::endl;
          _nTimeSteps = _volumeFiles.size();
          return _volumeFiles;
        }
      
      //assert(_nVolumeFiles == 1);
      assert(_volumeFiles.size()==1);
      
      std::vector<std::string> fnames;

      const std::string volumeFileName = _volumeFiles[/*fileName*/0];
      std::string marker_dollar("$");
      size_t startPos_dollar = volumeFileName.find(marker_dollar);
      size_t startPos_star = volumeFileName.find("*");

      const bool is_dollar = (startPos_dollar != std::string::npos);
      const bool is_star = (startPos_star != std::string::npos);
      
      //assert(!is_dollar || !is_star);
      
      if(!is_dollar && !is_star)
	{
	  fnames.push_back(volumeFileName);
	  _nTimeSteps = 1;
	}
      else if(is_dollar)
        {
          for(unsigned int timeStep=0; 
              timeStep<_nTimeSteps; timeStep+=_everyNthTimeStep)
            {
	      
              std::stringstream out;
              out << timeStep+_timeStepOffset;
              std::string numStr = out.str();
              while((int )numStr.size() < _numFixedLen)
                numStr = "0" + numStr;
        
              std::string fname(volumeFileName);
              
              fname.replace(startPos_dollar, marker_dollar.length(), numStr);
              
              //std::cout << "gen file name: " << fname << std::endl;
              fnames.push_back(fname);
            }
        }
      else if(is_star)
        {
	  //std::vector<std::string> results = call_cmd("ls " + volumeFileName);
	  const auto configSplitIdx = volumeFileName.find_last_of("/\\");
	  assert(configSplitIdx != std::string::npos);
	  const std::string cmdStr = "find " + volumeFileName.substr(0, configSplitIdx)+ " -maxdepth 1 -name \"" + volumeFileName.substr(configSplitIdx+1) + "\"";
	  std::cout << "cmdStr: " << cmdStr << std::endl;
	  std::vector<std::string> results = call_cmd(cmdStr);

	  std::sort(results.begin(), results.end());
	  
	  if(_nTimeSteps == 0)
	    _nTimeSteps = results.size();

	  assert(fnames.empty());

	  const size_t endT = std::min(results.size(), (size_t)_nTimeSteps);
	  for(size_t t=_timeStepOffset; t<endT; t+=_everyNthTimeStep)
	    {
	      if(false)
		std::cout << "add time step " << t << " " << endT << std::endl;
	      fnames.push_back(results[t]);
	    }
	  
	  // std::cout << __PRETTY_FUNCTION__
	  // << " fnames.size() " << fnames.size()
	  // << " getNTimeSteps() " << getNTimeSteps()
	  // << " _nTimeSteps " << _nTimeSteps
	  // << " _everyNthTimeStep " << _everyNthTimeStep
	  // << std::endl;
	  assert(fnames.size() == getNTimeSteps());
#if 0
	  if(_nTimeSteps == 0)
            {
              _nTimeSteps = fnames.size();
              std::cout << "found nTimeSteps=" << _nTimeSteps << std::endl;
            }
	  else
	    {
	      assert(_nTimeSteps <= fnames.size());
	      fnames.erase(fnames.begin()+_nTimeSteps, fnames.end());
	    }
#endif
	  /*
          fnames = std::vector<std::string>(fnames.begin()+_timeStepOffset,
                                                fnames.begin()+std::min(_timeStepOffset+_nTimeSteps,
                                                                        (unsigned int)fnames.size()));
	  */
        }

      /*
      for(const auto& f : fnames)
        {
          std::cout << __func__ << " fname: " << f << std::endl;
        }
      */
      return fnames;
    }

  size_t getNTimeSteps() const
  {    
    //Round a / b to nearest higher integer value
    auto iDivUp = [](size_t a, size_t b) { return (a % b != 0) ? (a / b + 1) : (a / b); };
    
    //return _nTimeSteps/_everyNthTimeStep;
    return iDivUp(_nTimeSteps-_timeStepOffset, _everyNthTimeStep);
  }

  size_t getNVoxels() const
  {
    return
    static_cast<size_t>(_volDim.x)
    *static_cast<size_t>(_volDim.y)
    *static_cast<size_t>(_volDim.z);
  }

  //application settings
  bool _render;
  //char** _volumeFiles;
  std::vector<std::string> _volumeFiles;
  //int _nVolumeFiles;
  char* _regionFileLoad;
  char* _regionFileSave;
  int3 _volDim;
  voldattype _volumeDataType;
  volformat _volumeFormat;
  int _width;
  int _height;
  int _volDataOffsetElems;
  float _maxDiffFloodFill;
  float _regionVoxelValueGroupRelativeRadius;
  float _regionVoxelRootVoxelChance;  
  int _numFixedLen;
  bool _sameFileTime;
  unsigned int _timeStepOffset;

  voldattoscalar _volDatToScalar;

  float3 _voxelSize;

  size_t _everyNthTimeStep;

  unsigned int _nTimeSteps;
  
  std::vector<std::string> _fileNameBuf;

  //protected:
  
 public:

  #include "UserData.cpp"
  
  //protected:
  //  void processLine(char* cline);
};

// #ifndef NO_FS3D
// #include "read_FS3D_impl.cpp"
// #endif

#include "volData.cpp"

#endif
