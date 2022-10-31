#ifndef __HUFFMAN__
#define __HUFFMAN__

#include <iostream>
#include <cassert>
#include <vector>
#include <set>
#include <bitset>
#include <map>
#include "huffman_helper.h"

template<typename Tin, typename Tout>
void huffman()
{
  std::vector<Tin> in;
  {
    bool success =
      readFile(in,
               "/Users/freysn/dev/reyars/grd/data/Bucky.raw"
               //"/Users/freysn/Desktop/ds14_scivis_0128_e4_dt04_0.0700"
               );
    assert(success);
  }
  std::cout << "read data with "
            << in.size()*sizeof(Tin)
            << " bytes" << std::endl;

  //for(size_t i=0; i<in.size() ;i++)
  //in[i]=0;
  
  const size_t nElems =
    std::numeric_limits<Tin>::max()-
                           std::numeric_limits<Tin>::min()
    +1;
  std::vector<size_t> freq(nElems, 0);

  assert(std::numeric_limits<Tin>::min() == 0);
  std::cout << "freq has " << freq.size() << " entires\n";

  for(size_t i=0; i<in.size(); i++)
    freq[in[i]]++;

  std::set<std::pair<size_t, size_t> > freq_ordered;
  for(size_t i=0; i<freq.size(); i++)
    {
      freq_ordered.insert(std::make_pair(freq[i],i));
    }
  /*
  for(size_t i=0; i<freq_ordered.size(); i++)
    {
      std::cout << freq_ordered[i].first
                << ": " <<  freq_ordered[i].second
                << std::endl;
    }
  */
  size_t newNodeValue = nElems-1;

  std::vector<std::pair<size_t,bool> >
    parentMap(2*nElems,
              std::make_pair(std::numeric_limits<size_t>::max(),
                             false));
  
  while(freq_ordered.size() > 1)
    {
      std::cout << "--- " << freq_ordered.size() << std::endl;
      std::pair<size_t, size_t> e0, e1;
      e0 = *freq_ordered.begin();
      freq_ordered.erase(freq_ordered.begin());
      e1 = *freq_ordered.begin();
      freq_ordered.erase(freq_ordered.begin());

      newNodeValue++;
      
      freq_ordered.insert(
                          std::make_pair(e0.first+e1.first,
                                         newNodeValue));

      assert(parentMap[e0.second].first ==
             std::numeric_limits<size_t>::max());
      assert(parentMap[e1.second].first ==
             std::numeric_limits<size_t>::max());
      
      parentMap[e0.second].first = newNodeValue;
      parentMap[e0.second].second = false;
      parentMap[e1.second].first = newNodeValue;
      parentMap[e0.second].second = true;
      
      
    }
  assert(freq_ordered.size()==1);
  const size_t rootId = newNodeValue;
  std::cout << "root id: " << rootId << std::endl;

  // there cannot be a real value at the root node
  assert(rootId >= nElems);
  
  //
  // collect codes
  //
  std::vector<std::pair<Tout, unsigned char> >
    huffmanCodes(nElems);
  for(size_t i=0; i<nElems; i++)
    {
      Tout code = 0;
      unsigned char codeCount = 0;
      size_t nextId = i;
      
      do
        {          
          size_t id = nextId;
          //std::cout << "id " << id << std::endl;          
          
          assert(id != std::numeric_limits<size_t>::max());
          assert(id<parentMap.size());
          nextId = parentMap[id].first;
          const bool right = parentMap[id].second;
          code = (code << 1) | right;
          codeCount++; 
        }
      while(nextId != rootId);
      assert(codeCount <= sizeof(Tout)*8);
      huffmanCodes[i].first = code;
      huffmanCodes[i].second = codeCount;
    }

  for(size_t i=0; i<nElems; i++)
        std::cout << "value: " << i
                  << " | code "
                  << std::bitset<16>(huffmanCodes[i].first)
                  << " | cnt " << (int) huffmanCodes[i].second
                << std::endl;

  //
  // ENCODE
  //
  std::vector<Tout> encoded((in.size()*sizeof(Tin))/sizeof(Tout));
  size_t bitCnt = 0;
  for(size_t i=0; i<in.size(); i++)
    {
      const Tin v = in[i];
      const Tout o = huffmanCodes[v].first;
      const Tout cnt = huffmanCodes[v].second;

      const size_t elemWidth = 8*sizeof(Tout);

      const unsigned int thisElem =
        bitCnt/elemWidth;
        
      const unsigned int offThisElem =
        bitCnt % elemWidth;
            
      encoded[thisElem] |= (o << offThisElem);

      const int spacePrev =
        elemWidth-offThisElem;

      if(cnt > spacePrev)
        encoded[thisElem+1] |= (o >> spacePrev);
      
      bitCnt+=cnt;
    }

  std::cout << in.size()*sizeof(Tin)
            << " byte input were reduced to "
            << bitCnt/8 << " bytes\n";


  //
  // DECODE
  //
  std::map<std::pair<Tout, unsigned char>,Tin> huffmanDecodes;
  for(size_t i=0; i<huffmanCodes.size(); i++)
    {
      assert(huffmanDecodes.find(huffmanCodes[i]) == huffmanDecodes.end());
      huffmanDecodes[huffmanCodes[i]] = i;
    }

  
  //std::vector<std::pair<Tout, unsigned char> >
  //huffmanCodes(nElems);


  
  {
    std::vector<Tin> in2(in.size());
    Tout candidate = 0;
    size_t candidateCount = 0;
    size_t bitCnt=0;
    size_t elemCnt=0;
    Tout elem=encoded[0];
    size_t elemOutId=0;
    
    while(true)
      {
        assert(candidateCount < sizeof(Tout)*8);
        //candidate = (candidate << 1) | (elem & 1);
        candidate |= ((elem & 1) << candidateCount);
        bitCnt++;
        candidateCount++;
        std::cout << "candidate: "
                  << std::bitset<16>(candidate)
                  << " " << bitCnt << std::endl;
        {
          typename std::map<std::pair<Tout, unsigned char>,Tin>::iterator it =
            huffmanDecodes.find(std::make_pair(candidate, candidateCount));
          if(it != huffmanDecodes.end())
            {
              std::cout << "decoded elem "
                        << "(" << std::bitset<16>(candidate) << ") "
                        << elemOutId << ": "
                        << (int)it->second << std::endl;
              in2[elemOutId] = it->second;
#ifndef NDEBUG
              {
                const Tin dec = in2[elemOutId];
                const Tin orig = in[elemOutId];                
                if(dec != orig)
                  {
                    std::cout << "dec: " << (int)dec
                              << " orig: " << (int)orig << std::endl;
                     std::cout << "enc: " << std::bitset<16>(huffmanCodes[orig].first)
                              << " " << (int) huffmanCodes[orig].second
                              << std::endl;
                  }
                assert(in2[elemOutId] == in[elemOutId]);
              }
#endif
              elemOutId++;

              /*
              std::cout << "next number to decode: " << (int) in[elemOutId] << std::endl;
              std::cout << "enc: " << std::bitset<16>(huffmanCodes[in[elemOutId]].first)
                              << " " << (int) huffmanCodes[in[elemOutId]].second
                              << std::endl;
              */
              candidate=0;
              candidateCount=0;

              if(elemOutId==in2.size())
                break;
            }
        } 
        const size_t elemWidth = 8*sizeof(Tout);
        
        elem = elem >> 1;

        if(bitCnt >= elemWidth)
          {
            bitCnt -= elemWidth;
            elemCnt++;
            assert(elemCnt < encoded.size());
            elem=encoded[elemCnt];
          }
      }
  }
}

#endif
