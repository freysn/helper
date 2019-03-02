#ifndef __HELPER__HOSTNAME__
#define __HELPER__HOSTNAME__

#include <unistd.h>

namespace helper
{
  std::string hostname()
    {
      char hostname[1024];  
      hostname[1023] = '\0';
      gethostname(hostname, 1023);
      return std::string(hostname);
    }
}


#endif //__HELPER__HOSTNAME__
