#ifndef __COMPRESSION_OMP__
#define __COMPRESSION_OMP__

#include "compression.h"
#include <cstring>
#include <vector>
#include <omp.h>

#ifdef __APPLE__
int omp_get_max_threads()
{
  return 1;
}
#endif

std::pair<size_t, size_t> bytesOffNum_omp(int t, int nT, size_t nBytes)
{
  std::pair<size_t, size_t> offNum;
  
  offNum.second = nBytes/nT;
  offNum.first = t*offNum.second;
  
  // last thread needs to work till the end
  if(t==nT-1)
    offNum.second = nBytes-offNum.first;

  return offNum;
}

void offsetsUncompressed_omp(std::vector<size_t>& outOff, 
                             const std::vector<std::pair<int,int> > osize)
{
  // prefix sum
  outOff.resize(osize.size()+1);
  for(int t=0; t<osize.size(); t++)
      outOff[t+1] = outOff[t] + osize[t].second;
}



size_t compress_omp(std::vector<std::pair<int,int> >& osize, 
                    char* out, size_t maxOSize, char* in, size_t isize)
{

  const int nThreads = osize.size();
  std::vector<std::vector<char> > compressed(nThreads, 
                                             std::vector<char>(isize/nThreads*2));
  
#pragma omp parallel for
  for(int t=0; t<nThreads; t++)
    {
      //std::cout << "t: " << omp_get_thread_num() << std::endl;
      std::pair<size_t, size_t> offNum = bytesOffNum_omp(t, nThreads, isize);
      
      compress(osize[t].first, osize[t].second, 
               (char*) &compressed[t][0], 
               compressed[t].size()*2, 
               in+offNum.first,
               offNum.second);      
    }
    
  std::vector<size_t> outOff;
  offsetsUncompressed_omp(outOff, osize);

#pragma omp parallel for
  for(int t=0; t<nThreads; t++)
    {
      memcpy(out+outOff[t], (char*) &compressed[t][0], osize[t].second);
    }
  
  return outOff[nThreads];
}


void decompress_omp(unsigned char* out,
                    const std::vector<std::pair<int,int> >& osize, 
                    char* in, int decompressedSize/*, int nThreads*/)
{

  std::vector<size_t> outOff;
  offsetsUncompressed_omp(outOff, osize);

  const int nThreads = osize.size();
  
#pragma omp parallel for
  for(int t=0; t<nThreads; t++)
    {
      std::pair<size_t, size_t> offNum = bytesOffNum_omp(t, nThreads, decompressedSize);
      decompress(out+offNum.first,
                 osize[t].first, osize[t].second, 
                 in+outOff[t], offNum.second);
      
      //for(size_t i=0; i<offNum.second; i++)
      //  assert(*(((char*) &data[0])+offNum.first+i) == uncompressedVec[t][i]);

#if 0
      memcpy(((char*)(&uncompressed[0]))+offNum.first, 
             &uncompressedVec[t][0],
             offNum.second);
#endif
    }

}

#endif //__COMPRESSION_OMP__
