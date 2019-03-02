#define M_VEC
#include "../volData/vec.h"
#include "helper_cimg.h"

#include "../volData/volData.h"
#include "../volData/splitStr.h"
#include <map>
#include "../losslessCompression/bzip_helper/bzip_helper.h"

int main(int argc, char** argv)
{

  if(argc != 2)
    {
      std::cout << "exactly one input argument required\n";
      return 0;
    }
  const std::string inputFName(argv[1]);
  //std::vector<uchar4> img;
  std::vector<unsigned char> img;
  int nChannels = 0;
  V3<int> imgDim;

  
  {
    std::vector<unsigned char> tmp;
    helper::cimgRead(tmp, imgDim, nChannels, inputFName);
    img.resize(imgDim.x*imgDim.y*imgDim.z);
    std::memcpy(&img[0], &tmp[0], tmp.size());
  }

  std::cout << "dim: " << imgDim.x << " " << imgDim.y << " " << imgDim.z << " | # channels: " << nChannels << std::endl;

#if 0
  std::map<uchar4, size_t> colCnt;

  for(auto e : img)
    {
      auto it = colCnt.find(e);
      if(it != colCnt.end())
	it->second++;
      else
	colCnt.insert(std::make_pair(e, 1));
    }

  for(auto e : colCnt)
    std::cout << (int) e.first.x << " " << (int)e.first.y << " " << (int)e.first.z << " " << (int)e.first.w << " | " << e.second << std::endl;
#endif
  

  const V3<int> volDim = make_V3<int>(256, imgDim.y, imgDim.x);

  std::vector<unsigned char> vol(volDim.x*volDim.y*volDim.z, 0);

  {    
    std::map<V4<uint8_t>, unsigned char> col2scalar;
    /*
    col2scalar.insert(std::make_pair(make_uchar4(119, 185, 0, 255), 128));
    col2scalar.insert(std::make_pair(make_uchar4(0, 0, 0, 255), 255));
    */
    for(size_t z=0; z<volDim.z; z++)
      {
	//std::cout << "create vol " << z << std::endl;
	for(size_t y=0; y<volDim.y; y++)
	  for(size_t x=0; x<volDim.x; x++)
	  {
#if 0
	    auto it = col2scalar.find(img[i]);
	    if(col2scalar.empty())
	      vol[z*img.size()+i] = img[i];
	    else if(it != col2scalar.end())
	      vol[z*img.size()+i] = it->second;
#else
	    const size_t vol_x = x;
	    const size_t vol_y = y;
	    const size_t vol_z = z;

	    const size_t img_x = z;
	    const size_t img_y = y;

	    assert(img_x < imgDim.x);
	    assert(img_y < imgDim.y);

	    assert(vol_x < volDim.x);
	    assert(vol_y < volDim.y);
	    assert(vol_z < volDim.z);
	    auto v = img[img_x+img_y*imgDim.x];
	    v = 255-v;

	    v/=16;
	    vol[vol_x+volDim.x*(vol_y+volDim.y*vol_z)] = v;
#endif
	  }
      }
  }

  const std::string outputFName = split(inputFName, '.').front() + ".raw.bz2";

  std::cout << "volume resolution: " << volDim << std::endl;
  std::cout << "write volume to " << outputFName << std::endl;
  bzip_compress(outputFName, vol);
  //bzip_compress("cuda.raw.bz2", vol);
  //writeFile(vol, "nvidia.raw");
  
  return 0;
}
