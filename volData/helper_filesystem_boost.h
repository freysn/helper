#ifndef __HELPER_FILESYSTEM_BOOST__
#define __HELPER_FILESYSTEM_BOOST__

#include <boost/filesystem.hpp>

namespace helper
{
  void createDirectory_boost(const std::string& fname)
  {
    boost::filesystem::path rootPath (fname);
    boost::system::error_code returnedError;

    boost::filesystem::create_directories(rootPath, returnedError);
  }

  void removeDirectory_boost(const std::string& fname)
  {
    boost::filesystem::path rootPath (fname);
    boost::system::error_code returnedError;

    boost::filesystem::remove_all(rootPath, returnedError);
  }
};

#endif //__HELPER_FILESYSTEM_BOOST__
