#include "vec.h"
#include "VolDataHandlerCUDA.h"
#include "helper_idx.h"
#include "volData.h"


int main(int argc, char** argv)
{
  VolDataHandlerCUDA<unsigned char> vdh(argv[1]);

  auto dataIn = vdh[0];

  const int3 volDimIn = vdh.getVolDim();
  const int3 off = make_int3(256, 0, 0);
  //const int3 volDimOut = volDimIn-2*off;;
  const int3 volDimOut = make_int3(512, volDimIn.y, volDimIn.z);

  std::cout << "vol_dim_out: " << volDimOut.x << " " << volDimOut.y << " " << volDimOut.z << std::endl;
  

  std::vector<unsigned char> dataOut(volDimOut.x*volDimOut.y*volDimOut.z, 0);

  int3 p;
  int3 mid = make_int3(volDimOut.x/2, volDimOut.y/2, volDimOut.z/2);
  for(p.z=0; p.z<volDimOut.z; p.z++)
    for(p.y=0; p.y<volDimOut.y; p.y++)
      for(p.x=0; p.x<volDimOut.x; p.x++)
        {
          //int3 d = p-mid;          
          //if(sqrtf(d.x*d.x+d.z*d.z) < 320)
            {
              const auto idx0 = iii2i(p, volDimOut);
	      /*
              const auto idx1 = iii2i(make_int3(
                                                //volDimOut.x-1-p.x
                                                p.x
                                                ,
                                                volDimOut.y-1-p.y
                                                ,
                                                //p.z
                                                volDimOut.z-1-p.z
                                                ), volDimOut);
	      */
	      const auto idx1 = iii2i(p+off, volDimIn);
              dataOut[idx0] = dataIn[idx1];
            }
        }

  writeFile(&dataOut[0], volDimOut.x*volDimOut.y*volDimOut.z, argv[2]);
}
