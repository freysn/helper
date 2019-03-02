#ifndef __HELPER_REDIRECT_STREAM__
#define __HELPER_REDIRECT_STREAM__

#include <iostream>

namespace helper
{
  class RedirectStreamFile
  {
  public:
    bool start_in(const std::string fname)
    {
      if(cinbuf)
	return false;
      in.open(fname.c_str());
      cinbuf = std::cin.rdbuf(); //save old buf
      std::cin.rdbuf(in.rdbuf()); //redirect std::cin to fname
      return true;
    }

    bool start_out(const std::string fname)
    {/*
      if(coutbuf)
	return false;
      out.open(fname.c_str());
      coutbuf = std::cout.rdbuf(); //save old buf
      std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
     */
      freopen(fname.c_str(),"w",stdout);
      return true;
    }

    bool stop_in()
    {
      if(!cinbuf)
	return false;
      std::cin.rdbuf(cinbuf);   //reset to standard input again
      cinbuf = 0;
      return true;
    }

    bool stop_out()
    {
      if(!coutbuf)
	return false;
      std::cout.rdbuf(coutbuf); //reset to standard output again
      coutbuf = 0;
      return true;
    }
    
    std::streambuf *cinbuf=0;
    std::streambuf *coutbuf = 0;

    std::ofstream out;
    std::ifstream in;
  };

};

#endif // __HELPER_REDIRECT_STREAM__
