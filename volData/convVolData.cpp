#include "volData.h"


int main(int argc, char** argv)
{
  //m_uint3 fromDim(256,256,256);
  //char* filename = "../data/engine256.raw";

  //m_uint3 fromDim(512,512,1734);
  //char* filename = "/data/volumes/volume/vhp/visfemale/c_vf512x512x1734_uchar.raw";

  //m_uint3 fromDim(1024,1024,1080);
  //char* filename = "/data/volumes/volume/utct/veiled-chameleon.raw";
  
  //char* filename = "/homel/freysnl/data/volumes/porsche.raw";
  //m_uint3 fromDim(559, 1023, 347);

  //m_uint3 fromDim(1024, 1024, 515);
  //char* filename = "/data/volumes/volume/utct/ivory-billed-woodpecker_8bit.raw";

  //m_uint3 fromDim(1024, 1024, 885);
  //char* filename = "/data/volumes/volume/utct/malachite-kingfisher.raw";

  m_uint3 fromDim(1024, 1024, 975);
  char* filename = "/data/volumes/volume/utct/oldfield_mouse.raw";
  elemType_t elemType = USHORT;

  //char* filename = "/homel/freysnl/data/volumes/zeiss1024.raw";
  //m_uint3 fromDim(1024, 1024, 1024);
  
  //char* filename = "/data/volumes/volume/tuebingen/foot.raw";
  //m_uint3 fromDim(256, 256, 256);
  //elemType_t elemType = UCHAR;

  
  
  //elemType_t elemType = FLOAT;

  size_t nMaxElements = 1024*1024*256;
  char* filenameOut = "out.raw";

  std::cout << "resize dim\n";
  //m_uint3 toDim = resizeDim(fromDim, nMaxElements);
  m_uint3 toDim = resizeDim(m_uint3(fromDim.x, fromDim.y, fromDim.z), nMaxElements);
  
  std::cout << toDim.x << " " << toDim.y << " " << toDim.z << std::endl;

  std::cout << "load original data\n";
  

  unsigned short* fromData = loadRawFile(filename, fromDim.x*fromDim.y*fromDim.z, elemType);
      
  unsigned short* toData = new unsigned short[toDim.x*toDim.y*toDim.z];

  std::cout << "resample data\n";
  resample(toData, toDim, fromData, fromDim);
  

  float* toDataF = new float[toDim.x*toDim.y*toDim.z];
  for(size_t i=0; i<toDim.x*toDim.y*toDim.z; i++)
    {
      toDataF[i] = toData[i]/(float)0xffff;
    }
  
  std::cout << "write file\n";
  writeFile(toDataF, toDim.x*toDim.y*toDim.z, filenameOut);
  
  return 0;
}

