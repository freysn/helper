#ifndef __HELPER_ASCII_FILE__
#define __HELPER_ASCII_FILE__

#include <fstream>

#include <sstream>
#include "helper/helper_string.h"

namespace helper
{

  void assembleLine(std::string&)
    {      
    }
    
  template<typename T, typename... Targs>
   void assembleLine(std::string& s, T v, Targs... args)
    {      
    
    std::stringstream ss;
    ss <<v;
    if(s != "")
      s = s + " " + ss.str();
    else
      s = ss.str();

    assembleLine(s, args...);
    }

    template<typename... Targs>
    bool writeLine(const std::string& fname, bool newLine, bool append, Targs... args)
    {
      std::ios_base::openmode mode = std::ofstream::out;

    if(append)
      mode |= std::ofstream::app;
      
    std::ofstream f((fname).c_str(), mode);
    if(!f.is_open())
      return false;

    std::string s = "";
    assembleLine(s, args...);
    f << s;
    if(newLine)
      f << std::endl;
    return true;
    }
    

    
    
  template<typename V>
    bool writeASCIIv(const V& v, const std::string& fname, bool append=false)
  {
    std::ios_base::openmode mode = std::ofstream::out;

    if(append)
      mode |= std::ofstream::app;
      
    std::ofstream f((fname).c_str());
    if(!f.is_open())
      return false;
    bool first = true;
    for(const auto& e : v)
      {
	if(!first)
	  f << std::endl;
	f << e;
	first = false;
      }
    return true;
  }
  
template<typename V>
bool readASCIIv(V& v, const std::string& fname, bool trimNomitEmptyLines=false)
  {
    std::ifstream ifs(fname.c_str());
    if(!ifs.is_open())
      return false;
    std::string line;
    v.clear();
    while (std::getline(ifs, line))
      {
	if(trimNomitEmptyLines)
	  trim(line);

	if(trimNomitEmptyLines && line != std::string(""))
	  v.push_back(line);
      }
    return true;
  }
};

#endif //__HELPER_ASCII_FILE__
