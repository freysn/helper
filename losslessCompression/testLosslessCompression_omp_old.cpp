#include "compression_omp.h"
//#include "compression.h"
#include <cassert>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>


int main(int argc, char** argv)
{
  //omp_set_num_threads(8);
  
  const int nThreads = omp_get_max_threads();
  
  
  std::cout << "There are " << nThreads << " threads, " << omp_get_max_threads() << " max\n";

  typedef int T;

  std::vector<T> data;
  
  for(int i=0; i<100000; i++)
    data.push_back(1000*sin(i/1000.f));

  const size_t dataSizeBytes = data.size()*sizeof(T);
  
  std::vector<std::pair<int,int> > osize(nThreads);

  std::vector<std::vector<char> > compressed(nThreads, 
                                             std::vector<char>(dataSizeBytes*2));

#pragma omp parallel for
  for(int t=0; t<nThreads; t++)
    {
      std::pair<size_t, size_t> offNum = bytesOffNum_omp(t, nThreads, dataSizeBytes);
      
      compress(osize[t].first, osize[t].second, 
               (char*) &compressed[t][0], 
               compressed[t].size()*2, 
               ((char*) &data[0])+offNum.first,
               offNum.second);
    }

  size_t totalBytes = 0;
  for(int t=0; t<nThreads; t++)
    totalBytes += osize[t].second;

  std::cout << "compressed " << dataSizeBytes << " bytes into " << totalBytes << std::endl;

  std::vector<std::vector<char> > uncompressedVec(nThreads,
                                                  std::vector<char>(dataSizeBytes*2));
  
  
  std::vector<int> uncompressed(data.size(), 666);
#pragma omp parallel for
  for(int t=0; t<nThreads; t++)
    {
      std::pair<size_t, size_t> offNum = bytesOffNum_omp(t, nThreads, dataSizeBytes);
      decompress(/*(unsigned char*) &uncompressedVec[t][0], */
                 ((unsigned char*)(&uncompressed[0]))+offNum.first,
                 osize[t].first, osize[t].second, 
                 &compressed[t][0], offNum.second);

      //for(size_t i=0; i<offNum.second; i++)
      //  assert(*(((char*) &data[0])+offNum.first+i) == uncompressedVec[t][i]);

#if 0
      memcpy(((char*)(&uncompressed[0]))+offNum.first, 
             &uncompressedVec[t][0],
             offNum.second);
#endif
    }

  
  
  
  /*
  for(int t=0; t<nThreads; t++)
    {
      std::pair<size_t, size_t> offNum = bytesOffNum_omp(t, nThreads, dataSizeBytes);
      memcpy(((char*) &uncompressed[0])+offNum.first, 
             &uncompressedVec[t], 
             offNum.second);
    }
  */
  
  //decompress((unsigned char*) &uncompressed[0], osize0, osize1, &compressed[0], data.size()*sizeof(int));

  assert(uncompressed.size() == data.size());
  
  for(size_t i=0; i<data.size(); i++)
    {
      //std::cout << i << " " << uncompressed[i] << " " << data[i] << std::endl;
      assert(uncompressed[i]==data[i]);
    }                           

  return 0;
}
