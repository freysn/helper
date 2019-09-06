#ifndef __HELPER_READ_FILE2__
#define __HELPER_READ_FILE2__

#ifndef NO_CIMG
#include "helper/helper_cimg.h"
#endif
//#include "read_FS3D_impl.h"


#include <vector>
#include <string>
#include <limits>
#include <fstream>

namespace helper
{

  template<typename V>
    size_t readFile2(std::vector<V>& buf, std::string volumeFileName, size_t offsetElems=0, size_t nElems=std::numeric_limits<size_t>::max())
    {
#ifdef VERBOSE
      std::cout << "Open File: " << volumeFileName << std::endl;
#endif
      std::ifstream file(volumeFileName.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

      if(!file.is_open())
	{	  
	  return 0;
	}
  
      size_t size = file.tellg();

      if(nElems == std::numeric_limits<size_t>::max())
	nElems = size/sizeof(V) - offsetElems;

      if(!((offsetElems+nElems)*sizeof(V) <= size))
	std::cerr << __func__ << " expected " << (offsetElems+nElems)*sizeof(V) << " vs data size " << size << std::endl;
      assert((offsetElems+nElems)*sizeof(V) <= size);
  
  
      file.seekg (sizeof(V)*offsetElems, std::ios::beg);
      buf.resize(nElems);
      //assert(buf.size() * sizeof(V) == size);
      file.read((char*)&buf[0], nElems*sizeof(V));
      
      return buf.size();
    }


  

  template<typename T>
  bool readFile2(const std::vector<std::vector<T>>& vecv,
		 const std::string fname)
  {
    std::cout << "READ FILE VECTOR OF VECTORS\n";

    std::vector<uint8_t> buf;
    if(!readFile2(buf, fname))
      return false;

    uint8_t* p = &buf[0];

    while(p < &buf[0]+buf.size())
      {
	const size_t n = *reinterpret_cast<size_t*>(p);

	p+=sizeof(size_t);

	const auto e = reinterpret_cast<T*>(p)+n;
	
	vecv.emplace_back(reinterpret_cast<T*>(p), e);

	p=reinterpret_cast<uint8_t*>(e);
      }
    assert(p==&buf[0]+buf.size());
    
    return true;
  }
};

#endif //__HELPER_READ_FILE__
