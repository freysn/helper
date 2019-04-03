#include <iostream>
#include "downsample.h"
#include <random>
#include <helper_math.h>

int main(int argc, char** argv)
{
  int3 dim = make_int3(32, 32, 1);

  typedef std::mt19937 MyRNG;
  uint32_t seed_val=123;
  
  MyRNG rng;
  rng.seed(seed_val);
  std::uniform_int_distribution<int> uint_dist(0,255);

  std::vector<unsigned char> in(dim.x*dim.y*dim.z, 0);
  for(size_t z=0; z<dim.z; z++)    
    for(size_t y=0; y<dim.y; y++)
      for(size_t x=0; x<dim.x; x++)
        in[x+dim.x*(y+dim.y*z)] = x;

          //for(auto& e : in)
          //e = uint_dist(rng);

  std::vector<unsigned char> out;

  int3 outDim;
  
  reduceLinear<double>(out, outDim, in, dim, 4);


  auto printVol = [](std::vector<unsigned char> data, int3 dim)
    {
      for(size_t z=0; z<dim.z; z++)    
        for(size_t y=0; y<dim.y; y++)
          {
            for(size_t x=0; x<dim.x; x++)
              std::cout << (int) data[x+dim.x*(y+dim.y*z)] << " ";
            std::cout << std::endl;
          }      
    };

  std::cout << "in: " << dim.x << " " << dim.y << " " << dim.z << "\n";
  printVol(in, dim);

  std::cout << "out: " << outDim.x << " " << outDim.y << " " << outDim.z << "\n";
  printVol(out, outDim);
  
  
  return 0;
}
