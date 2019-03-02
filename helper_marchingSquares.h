#include <type_traits>
#include <cassert>
#include <tuple>
#include <limits>
#include "helper/helper_util.h"

#ifndef __HELPER__MARCHING_SQUARES__
#define __HELPER__MARCHING_SQUARES__

namespace helper
{
  
  /*           
    V3-E2- V2
     |	         |  
    E3	E1
     |            |
    V0-E0-V1
   */

  
  template<typename CODE>
  #ifdef __CUDACC__
  __device__ __host__
#endif
  int getEdgeCount(CODE code)
  {
    return
      //#ifndef __CUDACC__
#ifndef  __CUDA_ARCH__
      __builtin_popcount(code)
#else
      __popc(code)
#endif
      ;
  }


  template<typename CODE>
#ifdef __CUDACC__
  __device__ __host__
#endif
  int code2id(CODE code)
  {
    if(code & 1)
      return 0;
    else if(code & 2)
      return 1;
    else if(code & 4)
      return 2;
    else if(code & 8)
      return 3;
    else
      assert(false);
    return -1;
  }
  
  template<typename P, typename I>
  #ifdef __CUDACC__
  __device__ __host__
#endif
  P getVCodeOffset(const I code)
  {
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral");
#endif
    switch(code)
      {
      case 1:
	return P(1,0);
      case 2:
	return P(1,1);
      case 3:
	return P(0,1);
      default:
	return P(0,0);
      }
  }

  template<typename P, typename I>
  #ifdef __CUDACC__
  __device__ __host__
#endif
  P getECodeOffset(const I code)
  {
    assert(getEdgeCount(code)==1);
  #ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral");
  #endif
    switch(code)
      {
      case (1<<0):
	return P(0, -1);
      case (1<<1):
	  return P(1, 0);
      case (1<<2):
	return P(0,1);
      case (1<<3):
	return P(-1, 0);
      default:
	assert(false);
	return P(-1, -1);
      }
  }

  template<typename I>
#ifdef __CUDACC__
  __device__ __host__
#endif
  I getEOppositeCode(const I code)
  {
    
    assert(helper::getEdgeCount(code) == 1);
    
#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral");
#endif
    
    switch(code)
      {
      case (1<<0):
	return (1<<2);
      case (1<<1):
	  return (1<<3);
      case (1<<2):
	return (1<<0);
      case (1<<3):
	return (1<<1);
      default:
	assert(false);
	return -1;
      }
  }

  template<typename F, typename I>
  #ifdef __CUDACC__
  __device__ __host__
#endif
  V2<F> getEIntersectPos(const V4<F> t, const I code)
  {
    assert(getEdgeCount(code)==1);

#ifndef __CUDACC__
    static_assert(std::is_integral<I>(), "needs to be integral");
#endif    
    
    return
      ((code & 1) > 0) * V2<F>(t.x, 0) +
      ((code & 2) > 0) * V2<F>(1., t.y) +
      ((code & 4) > 0) * V2<F>(1.-t.z, 1.) +
      ((code & 8) > 0) * V2<F>(0., 1.-t.w);
  }

  template<typename T>
#ifdef __CUDACC__
  __device__ __host__
#endif
  auto resolveVertexValueEqIsovalue(T vertexValue, const T isovalue)
  {
    const auto m_eps = helper::eps<T>();

    if(helper::apprEq(vertexValue, isovalue, m_eps))
      vertexValue = isovalue + m_eps;

    return vertexValue;
  }
  
  template<typename P, typename V, typename TEXLOOKUP>
#ifdef __CUDACC__
  __device__ __host__
#endif
  auto marchingSquaresCell(P pos, const V isovalue, TEXLOOKUP texLookup)
  {
    const auto m_eps = helper::eps<V>();
    
    V4<V> rslt(-1., -1., -1., -1.);
      
    //auto va = texLookup(pos+getVCodeOffset<P>(0));

    uint32_t rsltCode = 0;

    V4<V> values
      (texLookup(pos+getVCodeOffset<P>(0)),
       texLookup(pos+getVCodeOffset<P>(1)),
       texLookup(pos+getVCodeOffset<P>(2)),
       texLookup(pos+getVCodeOffset<P>(3)));
    
    //std::cout << "values: " << values << " iso: " << isovalue << std::endl;

    for(int i=0; i<4; i++)      
      values[i] = resolveVertexValueEqIsovalue(values[i], isovalue);
	  //std::nexttoward(values[i],
	  //		  std::numeric_limits<V>::max());
    //isovalue + m_eps;
    
    
    for(int i=0; i<4; i++)
      {	
	//auto vb = texLookup(pos+getVCodeOffset<P>(i+1));
	const auto va = values[i];
	const auto vb = values[(i+1)%4];
	
	const auto tmp = (vb-va);
	if(std::abs(tmp) > m_eps)
	  {	  

	    const auto  t = (isovalue - va) / tmp;

	    assert(t != 0. && t != 1.);
	    const bool doesIntersect = (t > 0. && t < 1.);
	    

	    //assert(doesIntersect == ((va >= isovalue && vb <= isovalue) || (va <= isovalue && vb >= isovalue)));
	    assert(doesIntersect == ((va > isovalue) != (vb > isovalue)));

	    //std::cout << "MS, i: " << i << " v: " << va << "  " << vb << " t: "  << t << " doesIntersect: " << doesIntersect << std::endl;
	
	    if(doesIntersect)
	      {
		rslt[i] = t;
		rsltCode |= (1 << i);
	      }
	  }
	//va = vb;
      }

    return
      #ifdef __CUDACC__
      thrust::
      #else
      std::
      #endif
      make_tuple(rslt, rsltCode, values);
  }

  template<typename CODE, typename T>
#ifdef __CUDACC__
  __device__ __host__
#endif
  V2<CODE> isolineDecider(T tv)
  {
    //
    // ISOLINE DECIDER
    // Idea: investigate the order of intersection points either along x or y axis
    // Build pairs of first two and last two intersections
    //
    CODE codeA;
    CODE codeB;

    // edge 0 goes from left to right
    // edge 1 goes from right to left
    if(tv[0] < (1.-tv[2]))
      {
	codeA = 1+ 8;
	codeB = 4 + 2;
      }
    else
      {
	codeA = 1+ 2;
	codeB = 4 + 8;
      }
    return V2<CODE>(codeA, codeB);
  }

    template<typename CODE>
#ifdef __CUDACC__
  __device__ __host__
#endif
  CODE getFirstEdgeCode(CODE code)
  {
    CODE edgeCode;
    //
    // pick a starting edge code
    //
    if(code & 1)
      edgeCode=1;
    else if(code & 2)
      edgeCode=2;
    else if(code & 4)
      edgeCode=4;
    else if(code & 8)
      edgeCode=0;
    else
      assert(false);
    return edgeCode;
  }

  template<typename OP, typename P, typename V, typename TEXLOOKUP>
#ifdef __CUDACC__
  __device__ __host__
#endif
void walkIsoline(OP& op, P p, const V isovalue, TEXLOOKUP texLookup)
  {
    auto q = p;

    unsigned int lastEdgeCode = -1;

#ifndef NDEBUG
    unsigned int stepCount = 0;
#endif
    do
      {

#ifndef NDEBUG
	assert((++stepCount) < 100000);
#endif
	auto rslt = helper::marchingSquaresCell(q, isovalue, texLookup);

	auto tv =
#ifdef __CUDACC__
	  thrust::
#else
	  std::
#endif
	  get<0>(rslt);
	auto code =
	  #ifdef __CUDACC__
	  thrust::
#else
	  std::
#endif
	  get<1>(rslt);

	assert(code != 0 || p == q);

	if(!code)
	  break;

	//std::cout << "intersection at " << q << std::endl;

	const auto n = helper::getEdgeCount(code);

	assert(n==2 || n == 4);

	

	unsigned int edgeCode=0;
	    
	if(lastEdgeCode == static_cast<unsigned int>(-1))
	  {
	    edgeCode = helper::getFirstEdgeCode(code);
	  }
	else	      
	  {
	    edgeCode = helper::getEOppositeCode(lastEdgeCode);
	  }

	// this edge code must be present in current result code
	    
	assert(edgeCode & code);

	if(n==4)
	  {
	    auto codes = isolineDecider<decltype(code)>(tv);

	    if(codes.x & edgeCode)
	      code = codes.x;
	    else
	      code = codes.y;
	  }

	code -= edgeCode;

	assert(helper::getEdgeCount(code)==1);

	//you can find the position of rightmost set bit by doing bitwise xor of n and (n&(n-1) )

	const auto localOffset = helper::getEIntersectPos(tv, code);
	op((decltype(localOffset))(q) + localOffset);
	    
	q += helper::getECodeOffset<decltype(p)>(code);

	assert(helper::iiWithinBounds(q, texLookup.getDim(), decltype(texLookup.getDim())(-1, -1)));
	    
	lastEdgeCode = code;
	    
      }
    while(q != p);
  }
};

#endif // __HELPER__MARCHING_SQUARES__
