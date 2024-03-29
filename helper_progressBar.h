#ifndef __HELPER_PROGRESS_BAR__
#define __HELPER_PROGRESS_BAR__

#include <iostream>
#include "helper_ChronoTimer.h"

#ifndef NO_OMP
#include <omp.h>
#endif

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


  void progressBar_omp(double progress, const std::string message="", int barWidth=40)
  {
#ifndef NO_OMP
    if(omp_get_thread_num()==0)
#endif
      progressBar(progress, message, barWidth);
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
