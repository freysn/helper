#include "vec.h"
#include "openvdb_io.h"
#include "vec.h"
#include "VolDataHandlerCUDA.h"

int main(int argc, char** argv)
{
  std::string configFname;
  std::string vdbFName;
  if(argc != 3)
    {
      std::cout << "provide input config and output vdb name\n";
      return 0;
    }
  configFname = std::string(argv[1]);
  vdbFName = std::string(argv[2]);

  std::cout << "convert "
            << configFname
            << " to " << vdbFName
            << std::endl;
    
  
  VolDataHandlerCUDA<unsigned char> vdh(configFname);
  assert(vdh.getNTimeSteps() == 1);
  storeVDB(vdh[0], vdh.getVolDim(), vdbFName);
  
  return 0;
}
