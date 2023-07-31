#ifndef __HELPER_SIMPLE_TRANS_FUNC__
#define __HELPER_SIMPLE_TRANS_FUNC__

#include "helper_volData/splitStr.h"
#include <vector>

namespace helper
{
class SimpleTransFunc
  {
  public:

    SimpleTransFunc()
      {}
    
    SimpleTransFunc(std::string fname)
      {
        const bool success = read(fname);
        if(!success)
          {
            std::cerr << __func__
                      << " COULD NOT FIND " << fname << std::endl;
          }
        assert(success);
      }
    
    bool read(std::string fname)
    {
      std::ifstream infile(fname.c_str());
      if(!infile.is_open())
        return false;
      
      while (! infile.eof() )
        {
          std::string line;
          std::getline(infile,line);
          std::vector<std::string> strs = split(line, ' ');
          assert(strs.size()==4);

	  auto addValue = [](std::vector<unsigned char>& v, int a)
	  {
	    if(a<0 || a>=256)
	      {
		std::cout << __func__ << " value " << a << " not in acceptable range " << std::endl;
		exit(0);
	      }
	    v.push_back(a);
	  };

	  addValue(vr, std::stoi(strs[0]));
	  addValue(vg, std::stoi(strs[1]));
	  addValue(vb, std::stoi(strs[2]));
	  addValue(va, std::stoi(strs[3]));
	  
          //std::cout << "tf " << va.size() << " " << (int) va.back() << std::endl;
        }
      //assert(va.size()==256);
      return true;
    }
    
    static double lookupNorm_s(const double v, const std::vector<unsigned char>& va)
    {
      if(va.empty())
        return v;
      
      assert(v>=0. && v<=1.);
      double vi = v*(va.size()-1);

      int idx0 = vi;
      
      if(idx0<0)
        idx0=0;

      if(idx0 >= va.size()-1)
        idx0 = va.size()-2;      
      
      int idx1 = idx0+1;

      assert(idx0>=0);
      assert(idx1<va.size());

      const double out = ((vi-idx0)*va[idx1] + (idx1-vi)*va[idx0])/255.;

      /*
	if(v==1.)
        {
	std::cout << __func__ << " " << idx0 << " " << idx1 << " " << vi << " " << out << std::endl;
        }
      */
      return out;
    }
    
    double lookupOpacityMax(double v) const
    {
      return lookupNorm_s(v, va);
    }

    template<typename F4>
      void lookupRGBA(F4& rgba, double v) const
      {
	rgba.x = lookupNorm_s(v, vr);
	rgba.y = lookupNorm_s(v, vg);
	rgba.z = lookupNorm_s(v, vb);
	rgba.w = lookupNorm_s(v, va);
      }

    bool empty() const
    {
      return va.empty();
    }

    std::vector<unsigned char> get()
      {
        return va;
      }

  protected:
    std::vector<unsigned char> vr;
    std::vector<unsigned char> vg;
    std::vector<unsigned char> vb;
    std::vector<unsigned char> va;
  };
};

#endif // __HELPER_SIMPLE_TRANS_FUNC__
