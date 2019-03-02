#ifndef __HELPER_PROGRESS_BAR__
#define __HELPER_PROGRESS_BAR__

#include <iostream>
#include "helper/helper_ChronoTimer.h"
#include <sstream>

namespace helper
{
  void progressBar(double progress, const std::string message="", int barWidth=40)
  {
    std::stringstream ss;
    ss << message << " [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) ss << "=";
      else if (i == pos) ss << ">";
      else ss << " ";
    }
    ss << "] " << int(progress * 100.0) << " %";

    std::cout << ss.str() << "\r" << std::flush;

    if(progress == 1.)
      std::cout << std::endl;
  }

  struct Progress
  {
    Progress(size_t nIterations, std::string identfier="") : _identifier(identfier), _nIterations(nIterations)
    {}
    
    void print(size_t iteration)
    {
      const double progress = static_cast<double>(iteration) / _nIterations;
      double elapsed = _timer.get_s();
      double total = 0.;
	if(progress > 0)
	  total = 1./progress * elapsed;
      double remaining = total-elapsed;

      std::stringstream ss;
      ss << " | " << iteration << " of " << _nIterations << " | elapsed: " << elapsed << ", remaining: " << remaining << ", total: " << total;
      progressBar(progress, _identifier + ss.str(), 20);
    }
    
    helper::ChronoTimer _timer;
    std::string _identifier;
    size_t _nIterations;
  };
};

#endif //__HELPER_PROGRESS_BAR__
