#ifndef __HELPER_FILESYSTEM__
#define __HELPER_FILESYSTEM__

#ifndef NO_EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
#else
#include <sys/stat.h>
#endif
#include <fstream>

namespace helper
{
  template<typename PATH>
    bool createDirectories(const PATH& p)
    {
#ifndef NO_EXPERIMENTAL_FILESYSTEM
      //std::experimental::filesystem::error_code ec;
      std::error_code ec;
      const bool success = std::experimental::filesystem::create_directories(p, ec);
      return success;
#else
      return (mkdir(p.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) != -1;
#endif      
    }

  template<typename PATH>    
  std::uintmax_t removeAll(const PATH& p)
    {
#ifndef NO_EXPERIMENTAL_FILESYSTEM
      std::error_code ec;
      const auto n = std::experimental::filesystem::remove_all(p, ec);
      return n;
#else
      std::cerr << "WARNING: REMOVAL OF DIRECTORY ONLY IMPLEMENTED IN EXPERIMENTAL FILESYSTEM MODE\n";
      //#warning "REMOVAL OF DIRECTORY ONLY IMPLEMENTED IN EXPERIMENTAL FILESYSTEM MODE"
      return 0;
#endif
    }


  template<typename PATH>
    bool createCleanDirectory(const PATH& p)
  {
    removeAll(p);
    return createDirectories(p);
  }

  
  template<typename PATH>
  bool cleanDirectory(const PATH& p)
  {
    return createCleanDirectory(p);
  }

  bool fileExists(const std::string fname)
  {
    std::ifstream f(fname.c_str());
    return f.good();
  }
  
  auto fileExtensionPos(std::string fn)
  {
    return fn.find_last_of(".");
  }
  
  std::string fileExtension(std::string fn)
  {
    return fn.substr(fileExtensionPos(fn));
  }
  
  std::string fileName(std::string fn)
  {
    const auto pos = fn.find_last_of("/");
    if(pos==std::string::npos)
      return fn;
    else      
      return fn.substr(pos+1);
  }
  
  std::string fileNameNoExtension(std::string fn)
  {
    fn = fileName(fn);    
    
    return fn.substr(0, fileExtensionPos(fn));
  }
  
  
  bool hasExtension(std::string fn, std::string ext)
  {
    return fileExtension(fn)==ext;
  }
  
}

#endif //__HELPER_FILESYSTEM__
