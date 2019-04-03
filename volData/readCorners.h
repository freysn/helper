#ifndef __READ_CORNERS__
#define __READ_CORNERS__

#include "volData/splitStr.h"

template<typename T>
std::vector<T> readCorners(std::string fname)
{
  std::vector<T> markers;
  std::string line;
  std::ifstream myfile (fname.c_str());

  if (myfile.is_open())
    {
      while (! myfile.eof() )
        {
          getline (myfile,line);
          std::vector<std::string> str = split(line, ' ');
          if(str.empty())
            continue;
          assert(str.size()==3);
          T elem;
          elem.x = std::stoi(str[0]);
          elem.y = std::stoi(str[1]);
          elem.z = std::stoi(str[2]);
          markers.push_back(elem);
        }
      myfile.close();
    }
  return markers;
}

#endif //__READ_CORNERS__
