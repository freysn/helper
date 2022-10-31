#ifndef __HELPER_PIXEL_ART_SCALING__
#define __HELPER_PIXEL_ART_SCALING__

// https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#cite_note-25
// https://github.com/falichs/Depixelizing-Pixel-Art-on-GPUs

#include "helper_idx.h"

namespace helper
{
  template<typename T>
  V4<T> epx_px(const T& p, const V4<T>& n)
  {
    V4<T> o(p,p,p,p);
    
    if(n[2]==n[0] && n[2]!=n[3] && n[0]!=n[1]){o[0]=n[0];};
    if(n[0]==n[1] && n[0]!=n[2] && n[1]!=n[3]){o[1]=n[1];};
    if(n[3]==n[2] && n[3]!=n[1] && n[2]!=n[0]){o[2]=n[2];};
    if(n[1]==n[3] && n[1]!=n[0] && n[3]!=n[2]){o[3]=n[3];};

    return o;
  }

  template<typename IT, typename DIM>
  auto epx(const IT in, const DIM dim_in)
  {

    using T = typename std::remove_const   
      <typename std::remove_reference<decltype(in[0])>::type>::type;
    
    const DIM dim_out(2*dim_in);
    std::vector<T> out(helper::ii2n(dim_out));

    const auto n_in = helper::ii2n(dim_in);

    for(size_t i=0; i<n_in; i++)
      {
	const auto & v = in[i];
	V4<T> n(v,v,v,v);

	const auto p = helper::i2ii(i, dim_in);

	
	
	if(p.x > 0)
	  n[2]=in[helper::ii2i(DIM(p.x-1, p.y), dim_in)];
	if(p.x < dim_in.x-1)
	  n[1]=in[helper::ii2i(DIM(p.x+1, p.y), dim_in)];


	if(p.y > 0)
	  n[0]=in[helper::ii2i(DIM(p.x, p.y-1), dim_in)];

	if(p.y < dim_in.y-1)
	  n[3]=in[helper::ii2i(DIM(p.x, p.y+1), dim_in)];
	
	const auto px4 = epx_px(v, n);

	out[helper::ii2i(p*2, dim_out)]=px4[0];
	out[helper::ii2i(p*2+DIM(1,0), dim_out)]=px4[1];
	out[helper::ii2i(p*2+DIM(0,1), dim_out)]=px4[2];
	out[helper::ii2i(p*2+DIM(1,1), dim_out)]=px4[3];	
      }

    return std::make_tuple(out, dim_out);
  }
}




#endif //__HELPER_PIXEL_ART_SCALING__
