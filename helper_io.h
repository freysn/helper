#ifndef __HELPER_IO__
#define __HELPER_IO__

#include "helper/helper_readFile.h"
#include "helper/helper_writeFile.h"
#include "helper/helper_bzip.h"
#include "helper/helper_string.h"
#include "helper/helper_asciiFile.h"

namespace helper
{  
  bool is_txt(const std::string fname, const std::string postfix="")
  {
    const std::string key("txt");
    return postfix==key || fnameExt(fname)==key;
  }

  bool is_bzip(const std::string fname, const std::string postfix="")
  {
    const std::string key("bz2");
    return postfix==key || fnameExt(fname)==key;
  }

  template<typename T>
  bool readFileAuto(T& v, const std::string fname, const std::string postfix="")
  {
    bool success;
    if(is_bzip(fname, postfix))
      {
	std::cout << "bz2 decompress file " << fname << std::endl;
	success = helper::bzip_decompress(v, fname);
      }
    else if(is_txt(fname, postfix))
      {	
	std::cout << "read txt file " << fname << std::endl;
	std::vector<std::string> lines;
	success=readASCIIv(lines, fname,
			   true /*bool trimNomitEmptyLines*/);

	std::cout << "there are " << lines.size()
		  << " non-empty lines in txt file " << fname << std::endl;

	v.reserve(lines.size());

#if 0
	{
	  //std::array<char, 10> str{"42"};
	  std::string str { "12345678901234" };
	  int result;
	  if(auto [p, ec] = std::from_chars(str.data(), str.data()+str.size(), result);
	     ec == std::errc())
	    std::cout << result;
	}
#else	
	for(size_t i=0; i<lines.size(); i++)
	  {
	    typename std::remove_reference<decltype(v[0])>::type tmp;
	    // const std::string str=lines[i];

	    // double test;
	    // // const auto ec = std::get<1>
	    // //   (std::from_chars(str.data(), str.data()+str.size(), /*v[i]*/test));

	    // auto [p, ec] = std::from_chars(str.data(), str.data()+str.size(), /*v[i]*/test);	    
	    // hassertm(ec == std::errc(), str);

	    if(lines[i].front()=='#')
	      {
		std::cout << "omit commented line |"
			  << lines[i]
			  << "|\n";
		continue;
	      }
	    s2x(tmp, lines[i]);
	    v.push_back(tmp);
	    // if(auto [p, ec] = std::from_chars(str.data(), str.data()+str.size(), v[i]);
	  //    ec == std::errc())
	  //   std::cout << result;
	  // v.push_back();
	  }
#endif	
      }
    else
      {
	std::cout << "read raw file " << fname << std::endl;
	success = helper::readFile2(v, fname);
      }
    std::cout << "reading file " << fname << " was " << (success ? "" : " NOT ") << "successful\n";
    return success;
  }

  template<>
  bool readFileAuto(std::vector<std::string>& v, const std::string fname, const std::string postfix)
  {
    hassertm(is_txt(fname, postfix), "only txt files supported for strings\n");
      	
	std::cout << "read txt file " << fname << std::endl;
	return readASCIIv(v, fname,
			   true /*bool trimNomitEmptyLines*/);
  }


  template<>
  bool readFileAuto(std::vector<bool>& v,
		    const std::string fname,
		    const std::string postfix)
  {
    std::vector<uint8_t> v_byte;
    if(!readFileAuto(v_byte, fname, postfix))
      return false;

    v.resize(v_byte.size());
    std::copy(v_byte.begin(), v_byte.end(),
	      v.begin());
    return true;
  }

  template<>
  bool readFileAuto(std::vector<V2<int>>& v,
		    const std::string fname,
		    const std::string postfix)
  {
    std::vector<int> v_int;
    if(!readFileAuto(v_int, fname, postfix))
      return false;

    assert(v_int.size()%2==0);
    
    v.reserve(v_int.size()/2);

    for(size_t i=0; i<v_int.size(); i+=2)
      v.emplace_back(v_int[i], v_int[i+1]);
      
    return true;
  }

  template<typename T>
  bool writeFileAuto(const /*std::vector<T>*/T& v, const std::string fname)
  {
    if(is_bzip(fname))
      {
	std::cout << "bz2 decompress file " << fname << std::endl;
	return helper::bzip_compress(v, fname) > 0;
      }
    else if(is_txt(fname))
      {
#if 0
	assert(false);
	return false;
#else
	return helper::writeASCIIv(v, fname);
#endif
      }
    else
      {
	std::cout << "read raw file " << fname << std::endl;
	return helper::writeFile(v, fname);
      }
  }

  template<>
  bool writeFileAuto(const std::vector<bool>& v, const std::string fname)
  {
    return writeFileAuto
      (std::vector<uint8_t>(v.begin(), v.end()), fname);
  }

  template<>
  bool writeFileAuto(const std::string& v, const std::string fname)
  {
    std::ofstream myfile(fname.c_str());
    if (myfile.is_open())
      {
	myfile << v;	
	myfile.close();
	return true;
      }
    else
      return false;    
  }
}

#endif //__HELPER_IO__
