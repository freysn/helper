#ifndef __STR_TO_NUM__
#define __STR_TO_NUM__

#include <sstream>
#include <string>
#include <stdexcept>

template<typename T>
T strToNum(const std::string& numberAsString)
{
  T valor;

  std::stringstream stream(numberAsString);
  stream >> valor;
  if (stream.fail()) {
    std::runtime_error e(numberAsString);
    throw e;
  }
  return valor;
}

#endif //__STR_TO_NUM__
