#define M_VEC
#include "helper_readFile.h"
#include "helper_writeFile.h"
#include <cassert>
#include "volData/splitStr.h"
#include "volData/UserData.h"
#include "helper_readData.h"
#include "helper_string.h"

#if 0
template<typename T_OUT, typename T_IN>
void convert(std::string fnameOut, std::string fnameIn)
{  
  std::vector<T_IN> in;  
  helper::readRawFile(in, fnameIn);

  std::vector<T_OUT> out(in.begin(), in.end());
  helper::writeFile(out, fnameOut);  
  }

int main(int argc, const char** argv)
{
  if(argc !=3)
    {
      std::cout << "usage: [progname] [in] [out]\n";
      return 0;
    }
  convert<float, unsigned char>(argv[2], argv[1]);
  
  return 0;
}

#endif


int main(int argc, const char** argv)
{
  if(argc !=3)
    {
      std::cout << "usage: [progname] [in] [out]\n";
      return 0;
    }

  typedef float valueType_t;
  
  std::vector<std::vector<valueType_t>> data;

  UserData<V3<int>,V3<float>> ud;
  {
    const bool success = ud.readConfig(argv[1]);
    assert(success);
  }

  const std::vector<std::string> fileNames = ud.getFileNameBuf();
  
  helper::readData(data, ud/*, maxNElems*/);

  assert(data.size() == fileNames.size());
  
  valueType_t minv = std::numeric_limits<valueType_t>::max();
  valueType_t maxv = std::numeric_limits<valueType_t>::lowest();

  
  for(const auto &v : data)
    for(const auto &e : v)
      {
	minv = std::min(minv, e);
	maxv = std::max(maxv, e);
      }

  std::cout << "determined min/max: " << minv << " " << maxv << std::endl;
  assert(!data.empty());
  std::vector<uint8_t> tmp(data[0].size());
  for(size_t i=0; i<data.size(); i++)
    {
      for(size_t j=0; j<tmp.size(); j++)
	tmp[j] = 255*(data[i][j]-minv)/(maxv-minv);

      const double t = stod(split(fileNames[i], '/').back());

      const std::string fname(std::string(argv[2])+ "/" + helper::leadingZeros(static_cast<uint32_t>(1000000.*t),12));
      std::cout << "write " << fname << std::endl;
      helper::writeFile(tmp, fname);  
    }
  
  return 0;
}
