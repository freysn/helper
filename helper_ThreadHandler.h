#ifndef __HELPER_THREAD_HANDLER__
#define __HELPER_THREAD_HANDLER__

#include <thread>
#include <future>
#include <chrono>

namespace helper
{
  template<typename T>
    class helper_ThreadHandler
    {
    public:

      template<typename... Targs>
	void run(Targs... Fargs)
	{
	  _futureDiff = std::async(std::launch::async, Fargs...);
	}

      bool valid()
      {
	return _futureDiff.valid();
      }

      bool finished()
      {
	//waits for the result, returns if it is not available for the specified timeout duration 
	auto status = _futureDiff.wait_for(std::chrono::milliseconds(0));
	return (status == std::future_status::ready);
      }

      T get()
      {
	return _futureDiff.get();
      }

      void wait()
      {
	_futureDiff.wait();
      }
  
      std::future<T> _futureDiff;
    };

}

#endif //__HELPER_THREAD_HANDLER__
