#ifndef __HELPER_STRING__
#define __HELPER_STRING__

#include <iostream>
#include <iomanip>
#include <sstream>

#include <fstream>
#include <string>
#include <vector>
#include <sstream>


#include <algorithm> 
#include <cctype>
#include <locale>

#include <charconv>
#include "helper_assert.h"
#include "helper_volData/vec.h"

#include "helper_cmd.h"


namespace helper
{
  template<typename T>
    std::string leadingZeros(T a, size_t n)
    {
      std::stringstream ss;
      ss << std::setw(n) << std::setfill('0') << a;
      return ss.str();
    }

  std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

  // trim from start (in place)
  static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
      return !std::isspace(ch);
    }));
  }

  // trim from end (in place)
  static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
      return !std::isspace(ch);
    }).base(), s.end());
  }

  // trim from both ends (in place)
  static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
  }

  // auto trim = [](std::string s)
  // {
  //   if(s=="")
  //     return s;
	      
  //   const auto begin = s.find_first_not_of(" ");
  //   if(begin == std::string::npos)
  //     return std::string("");
	      
  //   const auto end = s.find_first_of(" ", begin);
  //   size_t len = std::string::npos;
  //   if(end != std::string::npos)
  //     len = end-begin;
  //   //std::cout << "|" << s << "|" << begin << "|" << end << "|" << len << "|\n";
  //   return s.substr(begin, len);
  // };

  // trim from start (copying)
  static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
  }

  // trim from end (copying)
  static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
  }

  std::string fnameExt(std::string fname)
  {
    {
      const auto pos = fname.find_last_of("/");

      if(pos != std::string::npos)
	{
	  assert(fname[pos]=='/');
	  fname=fname.substr(pos+1);
	}
    }

    const auto pos=fname.find_last_of(".");
    if(pos==std::string::npos)
      return std::string("");
    else
      return fname.substr(pos+1);

    // return
    //   (pos != std::string::npos) && fname.substr(pos+1) == "bz2";
  }


  
  template<typename T>
  bool s2x(T& o, const std::string& str)
  {
    auto [p, ec] = std::from_chars(str.data(), str.data()+str.size(), o);
    const bool success=(ec == std::errc());
    hassertm(success, str);
    return success;
  }

  template<>
  bool s2x<double>(double& o, const std::string& str)
  {
    o=stod(str);
    return true;
  }

  template<>
  bool s2x<float>(float& o, const std::string& str)
  {
    o=stof(str);
    return true;
  }

    template<>
    bool s2x<V4<uint8_t>>(V4<uint8_t>& /*o*/, const std::string& /*str*/)
    {
      assert(false);
      return false;
    }

  template<>
  bool s2x<V4<double>>(V4<double>& /*o*/, const std::string& /*str*/)
  {
    assert(false);
    return false;
  }

  template<>
  bool s2x<V2<float>>(V2<float>& /*o*/, const std::string& /*str*/)
  {
    assert(false);
    return false;
  }


  std::vector<std::string> genFileNames(const std::vector<std::string> _volumeFiles)
  {
    const size_t _everyNthTimeStep=1;
    size_t _nTimeSteps=0;
    const size_t _timeStepOffset=0;
    const size_t _numFixedLen=6;
      
    if(_volumeFiles.size() > 1)
      {
	std::cout << "found more than one file: ";
	for(auto v : _volumeFiles)
	  std::cout << " " << v;
	std::cout << std::endl;

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
#ifdef NO_CMD
	assert(false);
	exit(-1);
#else
	//std::vector<std::string> results = call_cmd("ls " + volumeFileName);
	const auto configSplitIdx = volumeFileName.find_last_of("/\\");
	assert(configSplitIdx != std::string::npos);
	const std::string cmdStr = "find " + volumeFileName.substr(0, configSplitIdx)+ " -maxdepth 1 -name \"" + volumeFileName.substr(configSplitIdx+1) + "\"";
	std::cout << "cmdStr: " << cmdStr << std::endl;
	  
	std::vector<std::string> results = helper::cmd(cmdStr);

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
#endif
      }
    return fnames;
  }
}

#endif //__HELPER_STRING__
