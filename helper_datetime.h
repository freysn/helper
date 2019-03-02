#ifndef __HELPER_DATETIME__
#define __HELPER_DATETIME__

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

namespace helper
{
  std::string getDateTimeStr()
    {
#if defined(__GNUC__) && (__GNUC__ < 5)
      return std::string("UNDEFINED");
#else
      auto t = std::time(nullptr);
      auto tm = *std::localtime(&t);

      std::ostringstream oss;
      oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
      auto str = oss.str();

      return str;
#endif
    }
};

#endif //__HELPER_DATETIME__
