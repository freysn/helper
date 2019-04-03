#ifndef __VOL_DATA_HANDLER_CUDA__
#define __VOL_DATA_HANDLER_CUDA__


#include <climits>
#include "UserData.h"
#include <cassert>
#include <fstream>
#include <iostream>
#if __cplusplus >= 201103L
#include <type_traits>
#endif

#include "helper_readFile.h"

template <typename U>         // in general case
struct single_channel
{
  using type = U; 
};
 
template <>         // for U = T*
struct single_channel<uchar4>
{
  using type = unsigned char;
};

#ifdef M_VEC
template <>         // for U = T*
struct single_channel<m_float3>
{
  using type = float;
};
#endif

template <typename W>
using SingleChannel = typename single_channel<W>::type;


  template<typename IN, typename OUT>
    void copyValue(OUT& out,
		   const IN& in)
  {
    out = in;
  }

#ifdef M_VEC
template<typename OUT>
void copyValue(OUT& out,
	       const V3<float>& in)
  {
    assert(false);
  }

template<typename IN>
void copyValue(V3<float>& out,
	       const IN& in)
  {
    assert(false);
  }
#endif
  

template<>
    void copyValue<uchar4, unsigned char>(unsigned char& out,
		   const uchar4& in)
  {
    //out = in;
    assert(false);
  }


template<>
    void copyValue<uchar4, double>(double& out,
		   const uchar4& in)
  {
    //out = in;
    assert(false);
  }

template<>
    void copyValue<uchar4, float>(float& out,
		   const uchar4& in)
  {
    //out = in;
    assert(false);
  }

template<>
void copyValue<unsigned char, uchar4>(uchar4& out,
		   const unsigned char& in)
  {
    //out = in;
    assert(false);
  }

template<>
void copyValue<unsigned short, uchar4>(uchar4& out,
		   const unsigned short& in)
  {
    //out = in;
    assert(false);
  }

template<>
void copyValue<float, uchar4>(uchar4& out,
		   const float& in)
  {
    //out = in;
    assert(false);
  }

//
// T is the type that data should be uploaded to the GPU
//
template<typename T>
class VolDataHandlerCUDA
{
public:
  //  template<typename T>
    class ret_t
    {
    public:
    ret_t() :
      data(0), timeStep(-1), changed(false)
      {};
    ret_t(T* idata, int itimeStep, bool ichanged) :
      data(idata), timeStep(itimeStep), changed(ichanged)
        {};
      
      T* data;
      int timeStep;
      bool changed;
    };
    
    void modifyConfig_volumeFiles(const std::string& s)
    {
      //assert(_userData._nVolumeFiles == 1);
      /*
      delete [] _userData._volumeFiles[0];      
      _userData._volumeFiles[0] = new char[s.size()+1];
      memcpy(_userData._volumeFiles[0], s.c_str(), s.size());
      _userData._volumeFiles[0][s.size()] = 0;
      */
      _userData._volumeFiles.resize(1);
      _userData._volumeFiles[0] = s;
      _volData.clear();
    }
    

    enum convMode_t
    {
      cm_direct,
      cm_value,
      cm_range
    };
 
  //
  // if reserveMaxNBytesCUDAMem=0, then reserve as much memory as it takes to store
  // the full data set
  //
    VolDataHandlerCUDA(std::string configFileName/*, size_t reserveMaxNBytesCUDAMem=0*/) :
  _timeStep(-1)
    {
      {
        const bool success = _userData.readConfig(configFileName.c_str());
        if(!success)
          {
            std::cout << "could not read config " << configFileName << std::endl;
            exit(-1);
          }
      }

      _init();
    }
  
 VolDataHandlerCUDA(UserData<> userData) :
  _timeStep(-1), _userData(userData)
    {
      _init();
    }

 protected:
  void _init()
  {
      _buf = std::vector<T>(_userData._volDim.x*_userData._volDim.y*_userData._volDim.z);

      _convMode = cm_direct;
#if __cplusplus >= 201103L && false
      _minMax = std::make_pair(std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::lowest());            

      const size_t hash_T = typeid(T).hash_code();

      std::cout << "input data type hash "  << _userData.getTypeHash() << " | requested data type hash: "<< hash_T << std::endl;

      if(_userData.getTypeHash() != hash_T || std::is_floating_point<T>())
        {
	  for(int t=0; t<getNTimeSteps(); t++)
	    {
	      std::vector<double> b;
	      readFile_copyValues_typeHash(b, getFName(t));
	      auto iters =  minmax_element(b.begin(), b.end());

	      const auto minv = *std::get<0>(iters);
	      const auto maxv = *std::get<1>(iters);
	      std::cout << "time step " << t << " minv: " << minv << " maxv: " << maxv << std::endl;
	      std::get<0>(_minMax) = std::min(std::get<0>(_minMax), minv);
	      std::get<1>(_minMax) = std::max(std::get<1>(_minMax), maxv);
	    }
	  
          //computeMinMax(_minMax);
          _convMode = cm_range;

	  //#ifdef VERBOSE
          std::cout << __func__ << ": min max " << _minMax.first << " " << _minMax.second << std::endl;
	  //#endif
          /*
          if(std::is_integral<T>())
            {
              _convMode = cm_range;              
            }
          */
        }
#endif
#ifdef VERBOSE
  std::cout << __func__ << ": conv mode: " << _convMode << std::endl;
#endif
    }

 public:


  template<typename IN, typename OUT>
    void copyValues(std::vector<OUT>& out,
                    const std::vector<IN>& in) const
  {
    out.resize(in.size());
    for(size_t i=0; i<in.size(); i++)
      copyValue(out[i], in[i]);
  }


  template<typename V, typename OUT>
    void readFile_copyValues(std::vector<OUT>& out,
                             std::string volumeFileName) const
  {
    std::vector<V> tmp;
    readFile(tmp, volumeFileName,
	     _userData._volDataOffsetElems,
	     getNVoxels());
    copyValues(out, tmp);
  }

  template<typename OUT>
    void readFile_copyValues_typeHash(std::vector<OUT>& out,
                                      std::string volumeFileName) const
  {
    //#ifndef NO_CPP11
#if __cplusplus >= 201103L
    if(_userData.getTypeHash() == typeid(float).hash_code())
      readFile_copyValues<float>(out, volumeFileName);
    else if(_userData.getTypeHash() == typeid(unsigned char).hash_code())
      readFile_copyValues<unsigned char>(out, volumeFileName);
    else if(_userData.getTypeHash() == typeid(uchar4).hash_code())
      {
	//readFile_copyValues<unsigned char>(out, volumeFileName);
	readFile_copyValues<uchar4>(out, volumeFileName);
      }
    else if(_userData.getTypeHash() == typeid(unsigned short).hash_code())
      readFile_copyValues<unsigned short>(out, volumeFileName);
    else if(_userData.getTypeHash() == typeid(float3).hash_code())
      {
        std::vector<float3> tmp;
        assert(_userData._volumeFormat == volformat_raw);
        /*
        readFile(tmp, volumeFileName,
        _userData._volDataOffsetElems,
                 getNVoxels());
        */
        helper::readRawFile(tmp, volumeFileName, _userData._volDataOffsetElems, getNVoxels());
        
        std::vector<float> tmp2(tmp.size());
        for(size_t i=0; i<tmp.size(); i++)
          {
            tmp2[i] = sqrtf(tmp[i].x*tmp[i].x+tmp[i].y*tmp[i].y+tmp[i].z*tmp[i].z);
          }
        
        copyValues(out, tmp2);
      }
    else
#endif
      assert(false);
  }

  template<typename V>
    size_t readFile(std::vector<V>& buf, std::string volumeFileName,		
		    size_t offsetElems=0, size_t nElems=std::numeric_limits<size_t>::max()) const
    {
      if(_userData._volumeFormat == volformat_raw)
	{
	  const size_t result =
	    helper::readRawFile(buf, volumeFileName, offsetElems, nElems);
	  /*
	  if(false)
	    for(size_t i=0; i<buf.size(); i++)
	      buf[i] = helper::reverseFloatEndianness(buf[i]);
	  */
	  return result;
	  
	}
#ifndef NO_BZIP2
      else if(_userData._volumeFormat == volformat_raw_bz2)
	{
          bzip_decompress(buf, volumeFileName);                    
	  return buf.size();
	}
#endif
      else if(_userData._volumeFormat == volformat_png)
	{	  
	  int3 volDim;
	  int nChannels;
	  //std::cout << "read image file: " << volumeFileName << std::endl;
#ifndef NO_CIMG
	  
	  std::vector<SingleChannel<T>> buf2;
	  //static_assert(std::is_same<SingleChannel<T>, unsigned char>::value, "retval must be unsigned char");
	  const auto expectNChannels = sizeof(T)/sizeof(SingleChannel<T>);
	  
	  helper::cimgRead(buf2, volDim, nChannels, volumeFileName, (expectNChannels == 1));

	  std::cout << "read #channels: " << nChannels << " (expected " << expectNChannels << ")" << std::endl;
	  assert(expectNChannels == nChannels);
	  buf.resize(buf2.size()/nChannels);
	  std::memcpy(&buf[0], &buf2[0], buf2.size());
#else
          assert(false);
#endif

	  assert(volDim.x == getVolDim().x);
	  assert(volDim.y == getVolDim().y);
	  assert(volDim.z == getVolDim().z);
	  
	  return buf.size();
	}
      else if(_userData._volumeFormat == volformat_fs3d)
	{
	  #ifdef NO_FS3D
	  assert(false);
	  #else
	  std::string fname = volumeFileName;
	  #if 1
	  const bool readDomain = true;
	  int res[3];
	  int numcomp;
	  std::vector<double> timestepvalues;
	  readDataInfo(fname, readDomain, res, 
		       numcomp, timestepvalues);
	  assert(numcomp == 1);

	  std::vector<float> data;

	  const size_t n = res[0]*res[1]*res[2];
	  buf.resize((sizeof(float)/sizeof(V))*n);

	  int subext[6];
	  for(int d=0; d<3; d++)
	    {
	      subext[2*d] = 0;
	      subext[2*d+1] = res[d]-1;
	    }
	  readData(fname, res, subext,
		   numcomp, (float*) &buf[0]);
	  return buf.size();
	  #else
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


	  const size_t n = resolution[0]*resolution[1]*resolution[2];
	  
	  buf.resize((sizeof(float)/sizeof(V))*n);
	  memcpy(&buf[0], data, sizeof(float)*n);
	  free(data);
	  return buf.size();
	  #endif
#endif
	}
      assert(false);
      return 0;
    }

  
  void toHostMemory()
  {
    const size_t temporalStride = 
      getVolDim().x*
      getVolDim().y*
      getVolDim().z;
    
    _volData.resize(getNTimeSteps()*temporalStride);

    std::cout << "Allocated " << _volData.size() << " Bytes for storing volume data in memory\n";

    for(int i=0; i<getNTimeSteps(); i++)
      {
        ret_t ret = getH(i, 0);
        assert(ret.changed);
        assert(ret.timeStep == i);        
        memcpy(&_volData[(size_t)i*(size_t)temporalStride], ret.data, 
               temporalStride*sizeof(T));
      }
    _timeStep = -1;
  }

  //
  // mode == -1: choose highest available automatically
  // mode == 0: from disk
  // mode == 1: from host memory
  // mode == 2: from cuda device memory
  //

  ret_t getH_mem(int timeStep) const
  {
    assert(!_volData.empty());
    int3 d = getVolDim();
    return ret_t(const_cast<T*>(&_volData[(size_t)timeStep*d.x*d.y*d.z]),
                 timeStep, true);
  }

  std::string getFName(int timeStep) const
    {
      //assert(timeStep < _userData.getFileNameBuf().size());
      //return _userData.getFileNameBuf()[timeStep];
      return _userData.getFileName(timeStep);
    }

  template<typename VV>
    void getTimeStepsH_c(VV& buf) const
    {
      buf.resize(getNTimeSteps());

      for(int t=0; t<getNTimeSteps(); t++)
        {
          std::cout << __func__ << ": load " << t << "/" << getNTimeSteps() << std::endl;
          buf[t].resize(getNVoxels());
          ret_t r = getH_c(t, 0, buf[t], INT_MAX);
          assert(r.changed);
        }
      std::cout << __func__ << " complete.\n";
    }

  template<typename VV>
    VV getTimeStepsH_c() const
    {
      VV buf;
      getTimeStepsH_c(buf);
      return buf;
    }

  template<typename V>
    ret_t getH_c(int timeStep, int mode, V& buf, int prevTimeStep) const
    {

      //std::cout << __PRETTY_FUNCTION__ << "nTimeSteps " << _userData._nTimeSteps << std::endl;
      //int newTimeStep = timeStep % _userData._nTimeSteps;
      int newTimeStep = timeStep % getNTimeSteps();
          
      
      if(newTimeStep == prevTimeStep)
        {
          return ret_t(0, prevTimeStep, false);
        }
#if 0
      std::string volumeFileName;
      if(_userData._volumeFiles.size()>1)
        volumeFileName = _userData._volumeFiles[newTimeStep];
      else
        //assert(_userData._volumeFiles.size() == 1);
        volumeFileName = _userData._volumeFiles[/*fileName*/0];
#endif

      std::string volumeFileName = _userData._volumeFiles[/*fileName*/0];

    if(mode == -1)
      {
        if(!_volData.empty())
          mode = 1;
        else
          mode = 0;
      }

    switch(mode)
      {
        // from disk
      case 0:
        {
          //if(_userData._nTimeSteps > 1/* && _userData._sameFileTime == false*/)
            {
#if 0
              std::stringstream out;
              out << timeStep;
              std::string numStr = out.str();
              while(numStr.size() < _userData._numFixedLen)
                numStr = "0" + numStr;
              
              std::string marker("$");
              size_t startPos = volumeFileName.find(marker);
              if(startPos == std::string::npos)  
                {
                  volumeFileName += numStr + ".raw";      
                }
              else
                {
                  volumeFileName.replace(startPos, marker.length(), numStr);
                }
#else
              volumeFileName = getFName(timeStep);
#endif
            }
                      
          
          if(_convMode == cm_direct)
            {
              readFile(buf, volumeFileName,
                          _userData._volDataOffsetElems,
                          getNVoxels());
            }
          else if(_convMode == cm_value)
            {            
              readFile_copyValues_typeHash(buf, volumeFileName);
            }
          else if(_convMode == cm_range)
            {
#if __cplusplus >= 201103L && 0
              std::vector<double> tmp;
              readFile_copyValues_typeHash(tmp, volumeFileName);
                            
              
              std::pair<T,T> cap;

              if(std::is_integral<T>())
                cap = std::make_pair
                  (std::numeric_limits<T>::min(),
                   std::numeric_limits<T>::max()
                   );
              else
                cap = std::make_pair(0.f, 1.f);
              
              //assert(_minMax.first < _minMax.second);
              //const double fac =
	      const auto fac = 
                (cap.second-cap.first)/
                (_minMax.second-_minMax.first);
              
              buf.resize(tmp.size());
              for(size_t i=0; i<tmp.size(); i++)
                {
                  auto v = (tmp[i]-_minMax.first)*fac;
                  buf[i] = std::max(cap.first, std::min(static_cast<T>(v), cap.second));
                }
#else
              assert(false);
#endif
            }

          
          return ret_t(&buf[0], newTimeStep, true);
        }
        //break;
        
        // from host memory
      case 1:
        {
          return getH_mem(newTimeStep);
          /*
          assert(!_volData.empty());
          int3 d = getVolDim();
          return ret_t(&_volData[(size_t)_timeStep*d.x*d.y*d.z], _timeStep, true);
          */
        }
        //break;

        // from cuda device memory
      case 2:
        // not implemented yet
        assert(false);
        break;
        //invalid mode
      default:        
        assert(false);                
      };

    return ret_t();
  }

   
  template<typename V>
    ret_t getH(int timeStep, int mode, V& buf)
  {    
    ret_t ret = getH_c(timeStep, mode, buf, _timeStep);
    _timeStep = ret.timeStep;
    return ret;
  }

  ret_t getH(int timeStep, int mode)
  {
    return getH(timeStep, mode, _buf);
  }

#if 0
  ret_t getH2(int mode)
  {
    assert(_userData._nVolumeFiles == 1);
    std::string volumeFileName = _userData._volumeFiles[0];

    if(mode == -1)
      {
        if(!_volData.empty())
          mode = 1;
        else
          mode = 0;
      }

    switch(mode)
      {
        // from disk
      case 0:
        {
          std::cout << "Open File H2: " << volumeFileName << std::endl;
          std::ifstream file (volumeFileName.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
          assert(file.is_open());
          size_t size = file.tellg();
          file.seekg (0, std::ios::beg);
          
          if(_convMode == cm_same)
            {              
              assert(_buf.size()*sizeof(T) == size);                        
              file.read((char*) &_buf[0], size);          
              
            }
          else if(_convMode == cm_uchar)
            {              
              const size_t nElems = size/sizeof(unsigned char);
              std::vector<unsigned char> tmp(nElems);
              file.read((char*) &tmp[0], size);
              assert(_buf.size() == tmp.size());
              for(size_t i=0; i<nElems; i++)
                _buf[i] = 256*tmp[i];
            }
          else
            assert(false);
    
          file.close();
          
          return ret_t(&_buf[0], -1, true);
        }
        break;
        
        // from host memory
      case 1:
        {
          assert(!_volData.empty());
          int3 d = getVolDim();
          return ret_t(&_volData[(size_t)_timeStep*d.x*d.y*d.z], _timeStep, true);
        }
        break;

        // from cuda device memory
      case 2:
        // not implemented yet
        assert(false);
        break;
        //invalid mode
      default:        
        assert(false);                
      };

    return ret_t();
  }
#endif

  size_t getNVoxels() const
  {
    return
      static_cast<size_t>(_userData._volDim.x)
      *static_cast<size_t>(_userData._volDim.y)
      *static_cast<size_t>(_userData._volDim.z);
  }
  
  int3 getVolDim() const
  {
    int3 volDim;
    volDim.x = _userData._volDim.x;
    volDim.y = _userData._volDim.y;
    volDim.z = _userData._volDim.z;

    return volDim;    
  }

  float3 getVoxelSize() const
  {
    float3 v;
    v.x = _userData._voxelSize.x;
    v.y = _userData._voxelSize.y;
    v.z = _userData._voxelSize.z;
    return v;
  }

  int getNTimeSteps() const
  {
    //std::cout << __PRETTY_FUNCTION__ << " nTimeSteps " << _userData._nTimeSteps << std::endl;
    return _userData.getNTimeSteps();
  }

  #if 0
  template<typename V>
    void computeMinMax(std::pair<V,V>& minMax) const
    {      
      V minv = std::numeric_limits<V>::max();
      V maxv = std::numeric_limits<V>::min();

      //std::vector<double> b;
      std::vector<T> b;
      for(int t=0; t<getNTimeSteps(); t++)
        {
          V minl = std::numeric_limits<V>::max();
          V maxl = std::numeric_limits<V>::min();
          //getH_c(t, 0, b, INT_MAX);
          readFile_copyValues_typeHash(b, getFName(t));
          
          for(size_t e=0; e<b.size(); e++)
            {
	      /*
	      if(std::abs(b[e]) > 1.e+20)
		{
		  std::cout << "warning: ignoring value " << b[e] << std::endl;
		  continue;
		}
	      */
	      V value;
	      copyValue(value, b[e]);
              minl = std::min(minl, value);
              maxl = std::max(maxl, value);
            }
          minv = std::min(minl, minv);
          maxv = std::max(maxl, maxv);
          std::cout << "t: " << t
                    << " min: " << minl
                    << " max: " << maxl
                    << "| total min: " << minv
                    << " total max: " << maxv
                    << std::endl;
        }
      minMax.first = minv;
      minMax.second = maxv;
    }
  #endif

  size_t size() const
  {
    return getNTimeSteps();
  }

  std::vector<T> operator[](int i) const
    {
      assert(i>= 0 && i<getNTimeSteps());
      std::vector<T> buf;

      if(_volData.empty())
	{
	  ret_t rslt = getH_c(i, 0, buf, INT_MAX);
	  assert(rslt.changed);
	}
      else
	{
	  std::cout << "READ FROM MEM\n";
	  const size_t nVoxels = getNVoxels();
	  const size_t start = i*nVoxels;
	  const size_t end = start+nVoxels;
	  assert(end <= _volData.size());
	  buf = std::vector<T>(_volData.begin()+start, _volData.begin()+end);
	}
      //std::cout << "buf size " << buf.size() << " vs # voxels " << getNVoxels() << std::endl;
      assert(buf.size() == getNVoxels());
      return buf;
    }

  UserData<> getUserData() const
  {
    return _userData;
  }

  void setMinMax(std::pair<double, double> minMax)
  {
    _minMax = minMax;
  }
  
 private:  
  
  UserData<> _userData;  
  int _timeStep;

  std::vector<T> _buf;
  std::vector<T> _volData;

  std::pair<double,double> _minMax;

  convMode_t _convMode;
};

#endif //__VOL_DATA_HANDLER_CUDA__
