#ifndef __HELPER_PROGRESS_BAR__
#define __HELPER_PROGRESS_BAR__

#include <iostream>
namespace helper
{
  void progressBar(double progress, const std::string message="", int barWidth=40)
  {

    std::cout << message << " [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) std::cout << "=";
      else if (i == pos) std::cout << ">";
      else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();

    if(progress == 1.)
      std::cout << std::endl;
  }
};

#endif //__HELPER_PROGRESS_BAR__
