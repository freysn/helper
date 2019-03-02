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

  template<typename T>
  std::string removeCharacters(std::string str, T chars)
  {
    str.erase(
	      std::remove_if(str.begin(), str.end(),
			     [chars](char chr)
			     {
			       for(const auto c : chars)
				 {
				   if(chr == c)
				     return true;
				 }
			       return false;
			     }
			     ),
	      str.end());
    return str;
  }
}

#endif //__HELPER_STRING__
