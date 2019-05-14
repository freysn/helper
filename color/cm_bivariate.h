#ifndef __CM_BIVARIATE__
#define __CM_BIVARIATE__


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

#endif //__CM_BIVARIATE__
