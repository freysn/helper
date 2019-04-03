#include "compression_omp.h"
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>

typedef std::chrono::high_resolution_clock high_resolution_clock;
typedef std::chrono::microseconds microseconds;

int main(int argc, char** argv)
{
  std::vector<int> data;
  
  for(int i=0; i<1000000; i++)
    data.push_back(1000*sin(i/1000.f));

  std::vector<std::pair<int,int> > osize(4);
  std::vector<char> compressed(data.size()*sizeof(int)*2);

  
  high_resolution_clock::time_point start = high_resolution_clock::now();

  
  
  size_t size = compress_omp(osize, (char*) &compressed[0], compressed.size()*2, (char*) &data[0], data.size()*sizeof(int));

  const double time_ms = 
    std::chrono::duration_cast<microseconds>(high_resolution_clock::now()-start).count()
    /1000.f;
  std::cout << "compressed " << data.size()*sizeof(int) 
            << " bytes into " << size 
            << "took ms " << time_ms
            << std::endl;
  

  std::vector<int> uncompressed(data.size()*2);

  decompress_omp((unsigned char*) &uncompressed[0], osize, &compressed[0], data.size()*sizeof(int));
  
  for(size_t i=0; i<data.size(); i++)
    {
      assert(uncompressed[i]==data[i]);
    }                           
  
  return 0;
}
