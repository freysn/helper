#include "resample.h"
#include "VolDataHandlerCUDA.h"
#include "volData.h"
#include <iomanip> 
#include <sstream>
#include <limits>

int main(int argc, char** argv)
{
  

  std::string fnameBase = "../../data/resample2/resample";

  std::string foutBase = "../../data/resample/base";

  for(size_t t=0; t<75; t++)
    {
      std::vector<std::pair<float3,float> > buf;
      std::cout << "t " << t << std::endl;
      std::stringstream num;
      num << std::setw(3) << std::setfill('0') << t;
      readBinFile(buf, fnameBase+num.str()+".raw");

      std::cout << "there are " << buf.size() << " data points\n";
      
      std::vector<float> outData;
      int3 dim;
      resample(outData, dim, buf, 256);
      
      writeFile(&outData[0], outData.size(), foutBase+num.str()+".raw");
    }  
  return 0;
}
