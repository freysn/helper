#ifndef __HELPER_WRITE_FILE__
#define __HELPER_WRITE_FILE__

#include <fstream>

namespace helper
{  
template<typename T>
bool writeFile(T* ptr, size_t nElements, char* filenameOut, bool append=false)
{
  FILE* pFile = fopen ( filenameOut , append ? "wba" : "wb" );
  if(pFile == NULL)
    return false;
  fwrite((void*)ptr, sizeof(T), nElements, pFile);
  fclose(pFile);
  return true;
  
  /*
  ofstream myFile;
  myFile.open ("out.raw", ios::out | ios::binary);
  myFile.write((char*)ptr, nElements*sizeof(T));
  myFile.close();
  */
}

template<typename T>
bool writeFile(T* ptr, size_t nElements,
               std::string filenameOut, bool append=false)
{
  /*
  FILE* pFile = fopen ( filenameOut , "wb" );
  fwrite((void*)ptr, sizeof(T), nElements, pFile);
  fclose(pFile);
  */
  
  std::ofstream myFile;
  auto flags = std::ios::out | std::ios::binary;
  if(append)
    flags |= std::ios::app;
  myFile.open (filenameOut.c_str(), flags);
  if(!myFile.is_open())
    return false;
  myFile.write((char*)ptr, nElements*sizeof(T));
  myFile.close();
  return true;
}

template<typename T>
bool writeFile(const std::vector<T>& vec,
               std::string filenameOut, bool append=false)
{
  return writeFile((T*)&vec[0], vec.size(), filenameOut, append);
}

  template<typename T>
  bool writeFile(const std::vector<std::vector<T>>& vecv,
		 std::string filenameOut, bool append=false)
  {
    std::cout << "WRITE FILE VECTOR OF VECTORS\n";
    bool first = true;
    for(const auto & vec : vecv)
      {
	{
	  const bool app = (!first) || append;
	  const size_t n = vec.size();
	  if(!writeFile(&n, 1, filenameOut, app))
	    return false;	  
	}

	if(!writeFile(vec, filenameOut, true))
	  return false;
      }
    return true;
  }


  

template<typename O, typename T, typename S>
  bool writeFileConv(T* ptr, size_t nElements,
                     S filenameOut,
                     T imin, T imax,
                     O omin, O omax)
{
  std::vector<O> v(nElements);
  for(size_t i=0; i<nElements; i++)
    {
      const double n =
        ((double)ptr[i]-(double)imin)/(double)imax;
      O c = omin+(omax-omin)*n;
      v[i] = std::max(omin, std::min(c, omax));
    }
    
    return writeFile(&v[0], nElements, filenameOut);
}

template<typename T>
bool writeFileStr(const T str,
               std::string filenameOut) 
  {
    std::ofstream outfile(filenameOut);
    if(!outfile.is_open())
      return false;
    
    outfile << str;

    return true;
  }
}

#endif //__HELPER_WRITE_FILE__
