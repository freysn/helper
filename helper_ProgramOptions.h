#ifndef __HELPER_PO__
#define __HELPER_PO__

#include <vector>
#include "volData/splitStr.h"
#include "helper/helper_lexicalCast.h"
#include <cassert>
#include "helper/helper_asciiFile.h"
#include "helper/helper_string.h"

namespace helper
{
  template<typename T=const char**>
  class helper_ProgramOptions
  {
  public:

  helper_ProgramOptions(int argc, T argv, size_t from=1) :
    _argc(argc), _argv(argv), _argi(from)
      {}

#if 0
    static std::string trim(std::string s)
    {
      if(s=="")
	return s;
	      
      const auto begin = s.find_first_not_of(" ");
      if(begin == std::string::npos)
	return std::string("");
	      
      const auto end = s.find_first_of(" ", begin);
      size_t len = std::string::npos;
      if(end != std::string::npos)
	len = end-begin;
      //std::cout << "|" << s << "|" << begin << "|" << end << "|" << len << "|\n";
      return s.substr(begin, len);
    }
#endif

    std::string nextStr()
      {
	assert(_argi <= _argc);
	if(_argi == _argc)
	  _s="";
	else
	  {
	    //std::cout << "argi " << _argi << " argc" << _argc << std::endl;
	    _s = std::string(_argv[_argi]);
	    trim(_s);
	    _argi++;
	  }
	return _s;
      }
    
    static bool s_c(std::string __s, std::string x, std::string m="")
    {

      // print help message
      if(m != "" && (__s == "-h" || __s == "--help"))
	std::cout << "help: " << x << " | " << m << std::endl;
	      
      std::vector<std::string> xv = split(x, ',');	      
	      
      for(auto e : xv)
	{
	  trim(e);
	  const bool match = (__s == e);

	  std::cout << __s << " vs " << e << " " << match << std::endl;
	  if(match)
	    return true;
	}
      return false;
    }

    bool c(std::string x, std::string m="") const
    {
      return s_c(_s, x, m);
    }

    std::string getS() const
      {
	return _s;
      }

    bool saveToFile(const std::string fname)
    {

      struct
      {
	std::vector<std::string> operator()(const std::vector<std::string>& in, int)
	{return in;}
	std::vector<std::string> operator()(const char** in, int argi)
	{
	  std::vector<std::string> out;
	  for(int i=0; i<argi; i++)
	    out.emplace_back(in[i]);
	    return out;
	}
      } convert;
      return helper::writeASCIIv(convert(_argv, _argi), fname);
    }

#if 0
    bool handleProgramOptions(int _argc, const char** _argv)
    {      
      
      for(size_t _argi=1; _argi < _argc; _argi++)
	{
	  
	  std::string __s = std::string(_argv[_argi]);

	  
	  if(c("-d, --datconfs", "data config file")) {configFNames.push_back(nextStr());}
	  else if(c("-t, --trans", "transfer function file")){transFuncFNames.push_back(nextStr());}
	  else if(c("-s, --steps")){timeSteps.push_back(helper::lexicalCast<int>(nextStr()));}
	  else if(c(", --maxValue")){helper::lexicalCast(maxValue, nextStr());}
	  else if(c("--nIntermediates", "number of output intermediates")){helper::lexicalCast(nIntermediates, nextStr());}
	  else if(c("--termTime"))
	    {
	      helper::lexicalCast(termTime, nextStr());
	    }
	  else if(c("--termImprovement"))
	    {
	      helper::lexicalCast(termImprovement, nextStr());
	    }
	  else if(c("--termIterations"))
	    {
	      helper::lexicalCast(termIterations, nextStr());
	    }
	  else if(c("--distancelimit"))
	    {
	      helper::lexicalCast(distanceLimit, nextStr());
	    }
	  else if(c("--preNormData"))
	    {preNormData=true;}
	  else if(c("-r,--reductionFactor", "reductionFactor"))
	    {reductionFactors.push_back(helper::lexicalCast<int>(nextStr()));}
	  else if(c("--selectSliceParams", "selectSlice"))
	    {selectSliceParams.push_back(helper::lexicalCast<double>(nextStr()));}
	  else if(c("--outDir", "output directory"))
	    {outDir = nextStr();}
	  else if(c("--alignCenters", "align centers"))
	    {alignCenters = true;}
	  else if(c("--doNotWriteResult", "do not write result"))
	    {doWriteResult = false;}
	  else if(c("--termDistRef"))
	    {termDistRef = nextStr();}
	  else if(c("--dataInputFormat"))
	    {dataInputFormat = static_cast<dataInputFormat_t>
		(helper::lexicalCast<int>(nextStr()));}
	  else if(c("--timeSteps"))
	    {timeSteps.push_back(helper::lexicalCast<size_t>(nextStr()));}
	  else if(c("-h, --help", "print help message")) {}
	  else
	    {
	      std::cerr << __PRETTY_FUNCTION__
			<< " argument not recognized: " << _argv[_argi] << std::endl;
	      assert(false);
	      exit(-1);
	    }
	}      
      return true;
    }
#endif

  private:
    int _argc;
    //const char** _argv;
    const T _argv;
    int _argi;
    std::string _s;
  };
}
#endif //# __HELPER_PO__
