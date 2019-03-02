#ifndef __CHRONO_TIMER__
#define __CHRONO_TIMER__

#include <chrono>
#include <cassert>

namespace helper
{
  class ChronoTimer
  {
    typedef std::chrono::time_point<std::chrono::system_clock> timePoint_t;
  public:
    ChronoTimer()
      {
	start();
      }
 
    void start()
    {
      init_start();
      acc_time = 0.;
      pauseActive=false;
    }

    void restart()
    {
      start();
    }

    static timePoint_t now()
    {
      return std::chrono::system_clock::now();
    }

    template<typename T>
      double get() const
      {
	assert(!pauseActive);
	timePoint_t tp_end = now();
	return std::chrono::duration_cast<T>(tp_end-tp_start).count();
      }


    double getTotal_s() const
    {
      assert(std::isfinite(acc_time));
      if(pauseActive)
	return acc_time;
      else
	return acc_time+get_s();
    }

    double get_s() const
    {
      assert(!pauseActive);
      return get_us_t()/1.e6;
    }

    double get_ms() const
    {
      return
	//acc_time*1.e3+
	get_us_t()/1.e3;
    }

    void pause()
    {
      assert(!pauseActive);
      assert(std::isfinite(acc_time));
      acc_time += get_s();
      assert(std::isfinite(acc_time));
      pauseActive=true;
      //init_start();
    }

    void unpause()
    {
      assert(std::isfinite(acc_time));
      assert(pauseActive);
      pauseActive=false;
      //assert(acc_time > 0.);
      init_start();
    }
 
    /*
      double get_s_t()
      {
      return get<std::chrono::seconds>();
      }

      double get_ms_t()
      {
      return get<std::chrono::milliseconds>();
      }
    */

 
    static std::string ctime()
    {
      const std::time_t t = std::chrono::system_clock::to_time_t(now());
      return std::string(std::ctime(&t));
    }
 
  private:

    void init_start()
    {
      tp_start = std::chrono::system_clock::now();
    }

    double get_us_t() const
    {
      return get<std::chrono::microseconds>();
    }

    bool pauseActive;
  
    timePoint_t tp_start;
    double acc_time;
  };
  /*
    template<typename T2>
    class ChronoTimerAccumulate : public ChronoTimer
    {
    public:
    ChronoTimerAccumulate() :
    ChronoTimer(),
    tacc(0.)
    {}

    void accumulate()
    {
    tacc += get<T2>();
    }

    double get_accumulate() const
    {
    return tacc;
    }

    private:
    double tacc;    
    };
  */
}
#endif //__CHRONO_TIMER__
