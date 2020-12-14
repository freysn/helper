#ifndef __HELPER__BZIP__
#define __HELPER__BZIP__

#include <cstdlib>
#include <cstring>

#include <bzlib.h>

#include <cassert>


#include <string>
#include <cstdio>
#include <vector>

#include <iostream>


// http://www.bzip.org/1.0.5/bzip2-manual-1.0.5.html#hl-interface

namespace helper
{
  template<typename T>
  std::pair<size_t,int> bzip_compress(FILE *f, const T* buf_in, size_t nElems)
  {
    int bzError;
    BZFILE *bzf;

    //The default value of 30 gives reasonable behaviour over a wide range of circumstances.

    //Parameter blockSize100k specifies the block size to be used for compression. It should be a value between 1 and 9 inclusive, and the actual block size used is 100000 x this figure. 9 gives the best compression but takes most memory.
    //..Allowable values range from 0 to 250 inclusive. 0 is a special case, equivalent to using the default value of 30.
    bzf = BZ2_bzWriteOpen(&bzError, f,
			  // blockSize100k
			  9,
			  0,
			  //work factor
			  0);

    assert(bzError == BZ_OK);

    {
      const size_t chunkElems =
	std::min((16*1024*1024)/sizeof(T), nElems);

      std::vector<T> buf(chunkElems);

      assert(chunkElems>=1);
      size_t nElemsTotal=0;
      for(size_t i=0; i<nElems; i+=chunkElems)
	{
	  const size_t nElemsThis = std::min(chunkElems, nElems-i);
	  std::copy(buf_in+i,
		    buf_in+i+nElemsThis,
		    buf.begin());

	  nElemsTotal +=nElemsThis;
	  const size_t nBytesThis= nElemsThis*sizeof(T);
	  BZ2_bzWrite (&bzError, bzf,
		       //buf+i,
		       &buf[0],
		       nBytesThis);
	  if(bzError != BZ_OK)
	    return std::make_pair(-1, bzError);
	}
      assert(nElemsTotal==nElems);
    }
    //assert(bzError == BZ_OK);


    unsigned int nBytesIn_lo32, nBytesIn_hi32, nBytesOut_lo32, nBytesOut_hi32;
    BZ2_bzWriteClose64(&bzError, bzf,
		       // abandon
		       0,
		       &nBytesIn_lo32,
		       &nBytesIn_hi32,
		       &nBytesOut_lo32,
		       &nBytesOut_hi32);

    


    const size_t nBytesOut = ((static_cast<size_t>(nBytesOut_hi32) << 32)
			      | static_cast<size_t>(nBytesOut_lo32));

#ifndef NDEBUG
    {
      const size_t nBytesIn = ((static_cast<size_t>(nBytesIn_hi32) << 32)
			       | static_cast<size_t>(nBytesIn_lo32));
      
      const bool eq = (nElems == nBytesIn/sizeof(T));
      if(!eq)
	{
	  std::cout << "ERROR: nElems " << nElems << " vs nBytesIn/sizeof(T) " << nBytesIn/sizeof(T) << " " << nBytesIn << " " << sizeof(T) << std::endl;
	  assert(eq);
	}
    }
#endif

    //assert(bzError == BZ_OK);
    if(bzError != BZ_OK)
      return std::make_pair(-1, bzError);
    return std::make_pair(nBytesOut, bzError);
  }

  template<typename T>
  int bzip_compress(std::string fname, T* buf, size_t nElems)
  {
    FILE* f;
    f = fopen(fname.c_str(), "wb");
    if (f == NULL) {
      //perror(fname);
      return -1;
    }

    fseek(f, 0, SEEK_SET);
    int nBytesOut, bzError;
    {
      std::pair<int,int> rslt = bzip_compress(f, buf, nElems);
      nBytesOut = rslt.first;
      bzError = rslt.second;
    }
    //std::tie(nBytesOut, bzError) = bzip_compress(f, buf, nElems);
    if (nBytesOut == -1 || bzError != BZ_OK)
      {
	std::cerr << __func__ << " failed to write " << nElems << " elems to " << fname << ", error code " << bzError << std::endl;
        assert(false);
	return -1;
      }

    fclose(f);
    return nBytesOut;
  }

  template<typename T>
  int bzip_compress(const std::vector<T>& buf, std::string fname)
  {
    // TODO: THIS IS NOT NICE, BUT WHY CAN'T I DO CONSTNESS?
    //auto buf2=buf;
    return bzip_compress(fname, &buf[0], buf.size());
  }

  template<typename T>
  int bzip_compress(std::string fname, const std::vector<T>& buf)
  {
    return bzip_compress(buf, fname);
  }

  template<typename T>
  bool bzip_decompress(std::vector<T>& out, FILE *f)
  {
    int bzError;
    BZFILE *bzf;

    bzf = BZ2_bzReadOpen(&bzError, f, 0, 0, NULL, 0);
    if (bzError != BZ_OK) {
      fprintf(stderr, "E: BZ2_bzReadOpen: %d\n", bzError);
      return false;
    }

    const size_t nElemsChunk = 1024;

    size_t off =0;
    while (bzError == BZ_OK)
      {
	out.resize(std::max(off+nElemsChunk, out.size()));
	const int nread = BZ2_bzRead(&bzError, bzf, &out[off],
				     nElemsChunk*sizeof(T));

	assert(nread % sizeof(T) == 0);

	/*
	  if (bzError == BZ_OK || bzError == BZ_STREAM_END)
	  {

	  size_t nwritten = fwrite(buf, 1, nread, stdout);
	  if (nwritten != (size_t) nread)
	  {
	  fprintf(stderr, "E: short write\n");
	  return -1;
	  }

	  }
	*/
	off += nread/sizeof(T);
      }
    out.resize(off);

    if (bzError != BZ_STREAM_END) {
      fprintf(stderr, "E: bzip error after read: %d\n", bzError);
      return false;
    }

    BZ2_bzReadClose(&bzError, bzf);
    return true;
  }

  template<typename T>
  bool bzip_decompress(std::vector<T>& out, std::string fname)
  {
    FILE* f;
    f = fopen(fname.c_str(), "rb");
    if (f == NULL) {
      //perror(fname);
      return false;
    }

    fseek(f, 0L, SEEK_END);
    size_t size = ftell(f);

    out.reserve(4*size/sizeof(T));

    fseek(f, 0, SEEK_SET);
    if (bzip_decompress(out, f) == -1)
      return false;

    fclose(f);
    return true;
  }
}

#endif //__BZIP_HELPER__
