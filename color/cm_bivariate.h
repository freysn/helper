#ifndef __CM_BIVARIATE__
#define __CM_BIVARIATE__


template<typename C>
auto cm_bi_josh(size_t nElemsPerDim)
{
  std::vector<C> m(nElemsPerDim*nElemsPerDim, C(0,0,0));
  
  C sourceCol(0.1, 0.1, 0.1);
  C targetCol0(0.6, 0.9, 0.7);
  C targetCol1(0.9, 0.65, 0.8);
  
  for(size_t i=0; i<nElemsPerDim; i++)
    {
      m[i] = helper::mix(sourceCol, targetCol0, static_cast<double>(i)/(nElemsPerDim-1));
      m[i*nElemsPerDim] = helper::mix(sourceCol, targetCol1, static_cast<double>(i)/(nElemsPerDim-1));
      
      // The Darken Blending Mode looks at the luminance values in each of the RGB channels and selects either the base color or blend color depending on which is darker.
      for(size_t y=0; y<nElemsPerDim; y++)
	for(size_t x=0; x<nElemsPerDim; x++)
	  {
	    const size_t idx = x+nElemsPerDim*y;
	    m[idx].x = std::min(m[x].x, m[y*nElemsPerDim].x);
	    m[idx].y = std::min(m[x].y, m[y*nElemsPerDim].y);
	    m[idx].z = std::min(m[x].z, m[y*nElemsPerDim].z);
	  }
    }
  
  return m;
}

#endif //__CM_BIVARIATE__
