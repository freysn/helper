#ifndef __HELPER_CMD__
#define __HELPER_CMD__

#include <algorithm>
#include <vector>
#include <cassert>

namespace helper
{
std::vector<std::string> cmd(const std::string command)
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

  std::vector<std::string> cmd_ls(const std::string fname)
  {
    return cmd("ls " + fname);
  }
}


#endif //__HELPER_CMD__
