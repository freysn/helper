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


std::tuple<std::vector<float>, V3<size_t>> readFileDMP(std::string fname)
{

  //A header with 3 integers of type uint16. They contain the size of the
  // x_1,x_2,x_3 size of the file, in a "x_1 fastest" configuration.

  V3<size_t> dim;
  {
    std::vector<uint16_t> buf;
    helper::readFile2(buf, fname, 0, 3);
    dim.x = buf[0];
    dim.y = buf[1];
    dim.z = buf[2];
  }
  std::cout << "dim: " << dim << std::endl;

  
  std::vector<char> buf;
  helper::readFile2(buf, fname, sizeof(uint16_t)*3);

  assert(buf.size() == dim.x*dim.y*sizeof(float));
  std::vector<float> buf2(dim.x*dim.y);
  std::memcpy(&buf2[0], &buf[0], buf.size());    
  
  return std::make_tuple(buf2, dim);  
}

std::tuple<std::vector<float>, V3<size_t>> readFileImg(std::string fname)
{

  //A header with 3 integers of type uint16. They contain the size of the
  // x_1,x_2,x_3 size of the file, in a "x_1 fastest" configuration.

  std::vector<uint8_t>  data;
  V3<size_t> dim;
  int nChannels;
  helper::cimgRead(data, dim, nChannels, fname);

  assert(nChannels == 1);
  
  std::cout << "dim: " << dim << std::endl;
  
  std::vector<float> buf2(data.begin(), data.end());
  assert(buf2.size() == data.size());
  
  return std::make_tuple(buf2, dim);  
}

std::tuple<std::vector<float>, V3<size_t>> readFileSelect(std::string fname)
{
  const std::string suffix = split(fname, '.').back();
  std::cout << "detected suffix " << suffix << std::endl;

  if(suffix == "DMP")
    return readFileDMP(fname);
  else
    return readFileImg(fname);
}

std::vector<std::string> genFileNames(const std::string volumeInFName, const std::string suffix)
{

  UserData<V3<int>,V3<float>> ud;

  if(suffix == "DMP")
    ud._volumeFiles.resize(1, volumeInFName+ "/rec_DMP_0/*.rec.DMP");
  else
    ud._volumeFiles.resize(1, volumeInFName+ "/*." + suffix);
  const std::vector<std::string> fileNames = ud.genFileNames();
  return fileNames;
}

int main(int argc, const char** argv)
{
  
  if(argc < 4)
    {
      std::cout << "usage: [progname] [out path] [suffix] [in0] [in1] [in2]\n";
      return 0;
    }

  std::vector<std::string> volumeInFNames(argv+3, argv+argc);

  typedef float valueType_t;
  typedef uint8_t outType_t;

  valueType_t minv = std::numeric_limits<valueType_t>::max();
  valueType_t maxv = std::numeric_limits<valueType_t>::lowest();

  V3<size_t> dim(0,0,0);
  
  for(const auto volumeInFName : volumeInFNames)
    {
      std::vector<std::vector<valueType_t>> data;

      const auto fileNames = genFileNames(volumeInFName, argv[2]);

      std::cout << "number of files: " << fileNames.size() << std::endl;

      if(fileNames.empty())
	{
	  std::cout << "no files found for " << volumeInFName << "\n";
	  return 0;
	}

      assert(dim.z == 0 || dim.z == fileNames.size());
      dim.z = fileNames.size();
      
      for(auto fname : fileNames)
	{
	  std::cout << fname << std::endl;
	  std::vector<float> buf;
	  V3<size_t> dimSlice;
	  std::tie(buf, dimSlice) = readFileSelect(fname);

	  assert(dim.x == 0 || dim.x == dimSlice.x);
	  assert(dim.y == 0 || dim.y == dimSlice.y);
	  
	  dim.x = dimSlice.x;
	  dim.y = dimSlice.y;
      
	  auto result = std::minmax_element(buf.begin(), buf.end());
	  minv = std::min(minv, *result.first);
	  maxv = std::max(maxv, *result.second);
	  std::cout << "current min/max: " << minv << " " << maxv << std::endl;
	}
    }

  std::cout << "determined min/max: " << minv << " " << maxv << std::endl;
  std::cout << "output dim: " << dim << std::endl;

  std::vector<outType_t> out(dim.x*dim.y*dim.z);

  for(const auto volumeInFName : volumeInFNames)
    {
      const auto fileNames = genFileNames(volumeInFName, argv[2]);
      size_t idx = 0;
      for(auto fname : fileNames)
	{
	  std::cout << fname << std::endl;
	  std::vector<float> buf;
	  V3<size_t> dim;
	  std::tie(buf, dim) = readFileSelect(fname);

	  for(auto e : buf)
	    {
	      assert(idx < out.size());
	      out[idx] = 255*(e-minv)/(maxv-minv);
	      idx++;
	    }
	}
    
      assert(idx == out.size());
      std::string fnameOut = std::string(argv[1]) + "/"+split(volumeInFName, '/').back();

      std::cout << "write output to " << fnameOut << std::endl;
      helper::writeFile(out, fnameOut + ".raw");

      UserData<V3<int>,V3<float>> ud;
      
      ud._volDim = dim;
      ud._volumeFiles.resize(1, fnameOut + ".raw");
      
      ud.writeConfig(fnameOut + ".config");      
    }

  return 0;
}
