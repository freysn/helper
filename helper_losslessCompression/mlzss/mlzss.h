#ifndef __MLZSS__
#define __MLZSS__

//nice description of algorithm: http://michael.dipperstein.com/lzss/

#ifndef __CUDACC__
#include <algorithm>
#endif


#include "mlzss_args.h"
#include "SharedBufHandler.h"


template<typename T, typename INFO, typename I, typename ARGS, bool dryRun/*=false*/>
#ifdef __CUDACC__
  __device__
#endif
  I lzss_encode(T* out_values,
                INFO* out_info,
                const T* in,
                const I in_size,
                const ARGS& args)
{
  
#ifndef __CUDACC__
  using namespace std;
  SharedBufHandler_dummy<T,I> sbh(in);
#else
  SharedBufHandler<T,I> sbh;
  //SharedBufHandler_dummy<T,I> sbh(in);
#endif

  sbh.load(in, buf_size);
  I i=0;
  I out_index=0;
  //
  // loop over input data sequence for encoding
  //
  while(i<in_size)
    {
      //std::cout << "------------\n";
      //std::cout << "i: " << i << std::endl;
      I longestMatch_len = 0;
      I longestMatch_off = 0;
      I j=0;
      const I maxLookahead = min(i, min(args.winSize,
                                        in_size-1-i));
      
      //maxLookahead = min(i, maxLookahead);

      //
      // find longest match
      //
#if 0
      {
        I k=0;
        while(j<maxLookahead)
          {
            const bool eq =
              (sbh.at(i+k) == sbh.at(i-maxLookahead+j+k));

            k+=eq;
            
            if(!eq || j+k==maxLookahead)
              {
                if(k > longestMatch_len)
                  {
                    longestMatch_len = k;
                    longestMatch_off = maxLookahead-j-1;
                  }

                k=0;
                j++;
              }
          }
      }
#else
      {
        const I start_from = i-maxLookahead;
        while(j<maxLookahead)
          {
            I k=0;
            {
              //const I start_from = i-maxLookahead+j;
              const I max_k = maxLookahead-j;
              const I base = start_from+j;
              while(k<max_k &&
                    //in[i+k] == in[i-maxLookahead+j+k]
                    sbh.at(i+k) == sbh.at(base+k)
                    )
                k++;
            }         
            if(k > longestMatch_len)
              {
                longestMatch_len = k;
                longestMatch_off = j;              
              }          
            j++;            
          }
        longestMatch_off = maxLookahead-longestMatch_off-1;
      }
#endif

      //std::cout << "longest len: " << longestMatch_len
      //<< std::endl;
      //for(int a=0; a<maxLookahead; a++)
      //std::cout << (int)in[i-maxLookahead+a] << " ";

      //std::cout << std::endl;
      //for(int a=0; a<maxLookahead; a++)
      //std::cout << (int)in[i+a] << " ";
      //std::cout << std::endl;
      
      const bool encoded = longestMatch_len >= args.minLen;

      I advance_nElems = 1;
      if(!dryRun)
        out_info[out_index] = encoded;
      if(encoded)
        {
          if(!dryRun)
            out_values[out_index] =
              (longestMatch_off+
               args.encodeFac*longestMatch_len);
          //i += longestMatch_len;
          advance_nElems = longestMatch_len;
        }      
      else
        {
          if(!dryRun)
            out_values[out_index] =
              //in[i];
              sbh.at(i);
          //i++;  
        }
      i += advance_nElems;

      const bool buf_update_required =
        sbh.end_idx < i+args.winSize;
      if((joint_warp_buf_update
          && i>=args.winSize 
          && __any(buf_update_required))
         || (!joint_warp_buf_update &&
             buf_update_required))
        sbh.load_from(in, i-args.winSize, in_size);
      //sbh.load_until(in, i+args.winSize);
      
      out_index++;
            
    }
  return out_index;
}


  template<typename T,
  typename INFO,
  typename I,  
  typename ARGS>
#ifdef __CUDACC__
  __device__
#endif
  I lzss_decode(T* out,
                const T* in_values,
                const INFO* in_info,
                const I in_size,                                                   
                const ARGS& args)
{
  I out_index = 0;
  for(I i=0; i<in_size; i++)
    {
      const I nBits = 8*sizeof(T);
      //const bool encoded = in_info[i];
      const bool encoded = (in_info[i/nBits] & (1 << (i%nBits)));
      if(encoded)
        {
          unsigned int len = in_values[i]/args.encodeFac;
          unsigned int off = in_values[i]-len*args.encodeFac;
          const I from = out_index-off-1;
          for(I j=from; j<from+len; j++)
            {
              out[out_index] = out[j];
              out_index++;
            }
        }
      else
        {
          out[out_index] = in_values[i];
          out_index++;
        }
    }
  return out_index;
}

#endif //__MLZSS__
