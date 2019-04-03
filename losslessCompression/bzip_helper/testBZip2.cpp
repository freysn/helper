#include "bzip_helper.h"
#include <iostream>
#include <ctime>


int main(int argc, char **argv)
{
  srand(time(NULL));

  std::vector<unsigned char> in, out;
  
  for(size_t i=0; i</*409600*/5*1024ul*1024ul*1024ul; i++)
    //in.push_back(/*rand()*/(i % 4)+2);
    in.push_back(rand()% 4);

  bzip_compress("tmp.bz2", in);
  bzip_decompress(out, "tmp.bz2");

  std::cout << "size: " << in.size() << " vs " << out.size() << std::endl; 
  assert(in.size() == out.size());
  
  for(size_t i=0; i<in.size(); i++)
    {
      if(in[i] != out[i])
        std::cout << i << " vs "
                  << (int)in[i] << " "
                  << (int)out[i] << std::endl;
      assert(in[i]==out[i]);
    }
  
  /*
  if (argc < 2)
    {
      fprintf(stderr, "usage: bunz <fname>\n");
      return EXIT_FAILURE;
    }
  return bunzip_one(argv[1]) != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
  */

  
}
