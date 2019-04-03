#include "mlzss.h"
#include "mlzss_std_args.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <ctime>


int main(int argc, char** argv)
{

  srand(time(NULL));
  
  std::vector<unsigned char> in, out_values, out_info, in_cmp;

  /*
  for(int i=0; i<20; i++)
    for(int j=0; j<i; j++)
      in.push_back(i);
  */

  for(int i=0; i<409600; i++)
    in.push_back(rand() % 4);
  
  out_values.resize(in.size(),255);
  out_info.resize(in.size(),255);

  lzss_args_t<unsigned int> args
    = mlzss_init_args_uchar();
  /*
  args.winSize = 15;
  args.minLen = 2;
  args.encodeFac =16;
  */
  
  unsigned int out_size =
    lzss_encode(&out_values[0], &out_info[0],
                &in[0], (unsigned int)in.size(), args);

  /*
  for(size_t i=0; i<in.size(); i++)
    std::cout << (int) in[i]<< ", ";
  std::cout << std::endl;

  for(size_t i=0; i<out_size; i++)
    {
      if(out_info[i])
        {
          unsigned int len = out_values[i]/args.encodeFac;
          unsigned int off = out_values[i]-len*args.encodeFac;
          std::cout << off << " " << len << std::endl;
        }
      else
        std::cout << (int) out_values[i]<< std::endl;
      
    }
  
  std::cout << std::endl;
  */
  std::cout << out_size << " vs " << in.size() << std::endl;


  in_cmp.resize(in.size(),255);

  size_t in_size =
    lzss_decode(&in_cmp[0],
                &out_values[0],
                &out_info[0],
                out_size,
                args);

  std::cout << in_size << " vs " << in.size() << std::endl;
  for(size_t i=0; i<in_size; i++)
    {
      //std::cout << (int) in_cmp[i]<< ", ";
      assert(in_cmp[i] == in[i]);
    }
  std::cout << std::endl;

  
  return 0;
}
