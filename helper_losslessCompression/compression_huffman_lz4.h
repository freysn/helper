#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <cassert>
#include "lz4/lz4.h"
#include "bcl/rle.h"
#include "bcl/huffman.h"
#include <iostream>
#include <vector>


bool compress_huffman_lz4(int& osize0, int& osize1, char* out, size_t maxOSize, char* in, size_t isize, char* tmp)
{  
  
  
  //Timer timer;
  //timer.start();
  
  //
  // DO HUFFMAN AT FIRST
  //
  
  
  //
  // RUN LENGTH ENCONDING BEFORE LZ4
  //
  /*
  unsigned int osize = RLE_Compress((unsigned char*) in, isize,
                                    (unsigned char*) out, maxOSize);
                                    assert(osize > 0);
  */

  

#if 1
  osize0 =  Huffman_Compress( (unsigned char*)in, (unsigned char*)tmp,
                              isize );
  

  //assert(LZ4_compressBound(isize) < (int) maxOSize);
  osize1 = LZ4_compress((char*) tmp, out, osize0);  
#else
  
  isize = LZ4_compress((char*) in, tmp, isize);
  
  assert(LZ4_compressBound(isize) < maxOSize);
  int osize =  Huffman_Compress( (unsigned char*)tmp, (unsigned char*)out,
                             isize );
  
#endif
  
  //std::cout << "Compression took: " << timer.elapsed_time() << std::endl;
  //timer.stop();
    
  return true;
}


bool compress_huffman_lz4(int& osize0, int& osize1, char* out, size_t maxOSize, char* in, size_t isize)
{
  char* tmp = new char[maxOSize];
  bool success = 
    compress_huffman_lz4(osize0, osize1, out, maxOSize, in, isize, tmp);
  delete [] tmp;
  return success;
}

bool compress_huffman_lz4(int& osize0, int& osize1, char* out, std::vector<char>& tmp, 
              char* in, size_t isize)
{
  return compress_huffman_lz4(osize0, osize1, out, tmp.size(), in, isize, &tmp[0]);
}

bool decompress_huffman_lz4(unsigned char* out, char* tmp, size_t outSize0, size_t /*outSize1*/, char* in, int decompressedSize)
{
  //Timer timer;
  //timer.start();
  
  //int inSize = 
  //LZ4_uncompress (in, tmp, outSize0);
  LZ4_decompress_fast(in, tmp, outSize0);
  
  Huffman_Uncompress( (unsigned char*)tmp, (unsigned char*)out,
                      outSize0, decompressedSize);

  //std::cout << "Decompression took: " << timer.elapsed_time() << std::endl;
  //timer.stop();
      
  return true;
}

 bool decompress_huffman_lz4(unsigned char* out, size_t outSize0, size_t outSize1, char* in, int decompressedSize)
{
  char* tmp = new char[outSize0];

  bool success = decompress_huffman_lz4(out, tmp, outSize0, outSize1, in, decompressedSize);
  
  delete [] tmp;
  
  return success;
}

#endif
