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
#include "helper/helper_assert.h"
#include "helper/volData/vec.h"


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

  // trim from both ends (copying)
  static inline std::string trim_copy(std::string s) {
    trim(s);
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
}

#endif //__HELPER_STRING__
