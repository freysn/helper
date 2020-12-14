#ifndef __CM_BIVARIATE__
#define __CM_BIVARIATE__

#include "helper/color/cm_map.h"

template<typename C>
auto cm_bi_josh(size_t nElemsPerDim)
{
  std::vector<C> m(nElemsPerDim*nElemsPerDim, C(0,0,0));
  
  //C sourceCol(0.95, 0.95, 0.95);
  C sourceCol(0.1, 0.1, 0.1);
  //C targetCol0(0.53, 0.87, 0.67);
  //C targetCol1(0.91, 0.64, 0.82);
  
  C targetCol1(235./255., 161./255., 3./255.);
  C targetCol0(0., 152./255., 1.);
    
  
  for(size_t i=0; i<nElemsPerDim; i++)
    {
      const double v = static_cast<double>(i)/(nElemsPerDim-1);
      m[i] = helper::mix(sourceCol, targetCol0, v);
      m[i*nElemsPerDim] = helper::mix(sourceCol, targetCol1,v);
      
      
      // The Darken Blending Mode looks at the luminance values in each of the RGB channels and selects either the base color or blend color depending on which is darker.
      // THIS IS ACTUALLY THE LIGHTEN MODE
      for(size_t y=1; y<nElemsPerDim; y++)
	for(size_t x=1; x<nElemsPerDim; x++)
	  {
	    const size_t idx = x+nElemsPerDim*y;
	    m[idx].x = std::max(m[x].x, m[y*nElemsPerDim].x);
	    m[idx].y = std::max(m[x].y, m[y*nElemsPerDim].y);
	    m[idx].z = std::max(m[x].z, m[y*nElemsPerDim].z);
	  }
    }
  
  return m;
}

void draw()
{
  //
  // TODO: JUST A DUMP FROM ANOTHER FILES, ADJUST!
  //
  using F = double;
  const size_t cm_nElemsPerDim=32;
  const auto biColMap = cm_bi_josh<V3<double>>(cm_nElemsPerDim);
  {
    const size_t n = 256;
    const size_t nBuckets =8;
    const size_t bucketSize = n/nBuckets;
    std::vector<V3<F>> cols(256*256);
    for(size_t y=0; y<n; y++)
      for(size_t x=0; x<n; x++)
	{
	  const V2<F> coords(
			     static_cast<F>(x-x%bucketSize)/(n-1),
			     static_cast<F>(y-y%bucketSize)/(n-1));
	  
	  cols[x+y*n] = cm_bi_map_norm(
				       coords, 
				       biColMap,
				       V2<size_t>(cm_nElemsPerDim, cm_nElemsPerDim));
	}
    
    // helper::cimgWriteNormRGB("colorMap.png", cols, 
    // 			     V3<size_t>(n, n, 1));
  }

}

#endif //__CM_BIVARIATE__
