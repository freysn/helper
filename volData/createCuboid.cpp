#include "vec.h"
#include "transform.h"
#include "cimg_helper.h"
#include "VolDataHandlerCUDA.h"
#include "splitStr.h"
#include "volData.h"

template<typename T, typename I3>
std::vector<T> run(std::vector<T>& in, I3 dim)
{
  std::vector<unsigned char> out;  

  out = transform(in, dim, [](double3 p)
                  {
                    //const double3 pcenter = make_double3(.5, .5, .5);

                    /*
                    p.x = p.x*p.x;
                    p.y = std::pow(p.y, 0.4);
                    p.z = p.z*p.z;
                    */
                    const double3 v =
                    make_double3((std::abs(p.x-0.5)*2),
                                 (std::abs(p.y-0.5)*2),
                                 (std::abs(p.z-0.5)*2));
                    
                    
                    p.x = std::pow(p.x, 0.6+v.y*1.4);
                    //p.y = std::pow(p.y, 0.75+v.z*1.11);
                    //p.z = std::pow(p.z, 0.7+v.x*1.105);
                    
                    return p;
                  }
                  );

  return out;
  
}


int main(int argc, char** argv)
{
  int3 dim=make_int3(64, 64, 64);

  std::vector<unsigned char> out(dim.x*dim.y*dim.z, 0);

  int3 p;
  for(p.z=8; p.z<36; p.z++)
    for(p.y=22; p.y<58; p.y++)
      for(p.x=3; p.x<42; p.x++)
        out[iii2i(p,dim)] = 255;
  
  writeFile(out,
            "cuboid_"
            + std::to_string(dim.x)
            + "_"+ std::to_string(dim.y)
            + "_"+ std::to_string(dim.z)
            + ".raw");
}
