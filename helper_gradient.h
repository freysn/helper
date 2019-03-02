#ifndef __HELPER_GRADIENT__
#define __HELPER_GRADIENT__

namespace helper
{
  template<typename TEXLOOKUP, typename POS>
#ifdef __CUDACC__
    __host__ __device__
#endif
  POS sobelFeldman2D(const TEXLOOKUP& texLookup, POS pos, const double delta)
  {
    POS out;    

    //const std::vector<double> weights = {3., 10., 3.};

    const V3<double> weights(3., 10., 3.);
  
    out.x =
      weights.x * texLookup(pos+POS(-delta, -delta))
      + weights.y * texLookup(pos+POS(-delta, 0))
      + weights.z * texLookup(pos+POS(-delta, delta))

      - weights.x * texLookup(pos+POS(delta, -delta))
      - weights.y * texLookup(pos+POS(delta, 0))
      - weights.z * texLookup(pos+POS(delta, delta))
      ;

    auto swap = [](POS pos)
      {
	POS out;
	out.x = pos.y;
	out.y = pos.x;
	return out;
      };

    out.y =
      weights.x * texLookup(pos+swap(POS(-delta, -delta)))
      + weights.y * texLookup(pos+swap(POS(-delta, 0)))
      + weights.z * texLookup(pos+swap(POS(-delta, delta)))

      - weights.x * texLookup(pos+swap(POS(delta, -delta)))
      - weights.y * texLookup(pos+swap(POS(delta, 0)))
      - weights.z * texLookup(pos+swap(POS(delta, delta)))
      ;

    // total weight = 64.
    return out/64.;
  }
  
  template<typename V, typename POS, typename DIM>
  POS sobelFeldman2D(const V& data, POS pos, DIM dim, const double delta=1.)
  {
    static_assert(POS::size == DIM::size, "POS and DIM should have same dimensionality");

    return
      sobelFeldman2D(
		     [&data, &dim](POS pos)
		     {return texLookupLinear3D(pos, data, dim);},
		     pos, delta
		     );
#if 0
    POS out;

    const std::vector<double> weights = {3., 10., 3.};
  
    out.x =
      weights[0] * texLookupLinear2D(pos+POS(-delta, -delta), data, dim)
      + weights[1] * texLookupLinear2D(pos+POS(-delta, 0), data, dim)
      + weights[2] * texLookupLinear2D(pos+POS(-delta, delta), data, dim)

      - weights[0] * texLookupLinear2D(pos+POS(delta, -delta), data, dim)
      - weights[1] * texLookupLinear2D(pos+POS(delta, 0), data, dim)
      - weights[2] * texLookupLinear2D(pos+POS(delta, delta), data, dim)
      ;

    auto swap = [](POS pos)
      {
	POS out;
	out.x = pos.y;
	out.y = pos.x;
	return out;
      };

    out.y =
      weights[0] * texLookupLinear2D(pos+swap(POS(-delta, -delta)), data, dim)
      + weights[1] * texLookupLinear2D(pos+swap(POS(-delta, 0)), data, dim)
      + weights[2] * texLookupLinear2D(pos+swap(POS(-delta, delta)), data, dim)

      - weights[0] * texLookupLinear2D(pos+swap(POS(delta, -delta)), data, dim)
      - weights[1] * texLookupLinear2D(pos+swap(POS(delta, 0)), data, dim)
      - weights[2] * texLookupLinear2D(pos+swap(POS(delta, delta)), data, dim)
      ;

    return out;
    #endif
  }









  template<typename TEXLOOKUP, typename POS>
#ifdef __CUDACC__
    __host__ __device__
#endif
  POS sobelFeldman3D(const TEXLOOKUP& texLookup, POS pos, const double delta)
  {
    POS out(0., 0., 0.);

    //const std::vector<double> weights = {3., 10., 3.};

    typedef double T;

    
    /*
        auto swapX = []
#ifdef __CUDACC__
__host__ __device__
#endif
      (double x, double y, double z)
      {return POS(z, x, y);};
    
    auto swapY = []
#ifdef __CUDACC__
      __host__ __device__
#endif
      (double x, double y, double z)
      {return POS(x,z, y);};
    
    auto swapZ = []
#ifdef __CUDACC__
__host__ __device__
#endif
      (double x, double y, double z)
      {return POS(x,y,z);};
    */
    
    auto processDir = [delta, texLookup, pos]
#ifdef __CUDACC__
      __host__ __device__
#endif
    //(decltype(swapX) op)
      (uint8_t mode)
      {

	struct
	{
	  //template<typename T>
#ifdef __CUDACC__
	  __host__ __device__
#endif
	  POS operator()(T x, T y, T z) const
	  {
	    if(mode==0)
	      return POS(z, x, y);
	    else if(mode==1)
	      return POS(x,z, y);
	    else
	      return POS(x,y,z);
	  }

	  uint8_t mode=0;
	} op;
	    
	op.mode = mode;
	const V3<double> weights(3., 10., 3.);
	double v = 0.;
	/*
	v = 0.*texLookup(pos);

	v += 0.*texLookup(op(-delta, -delta, -delta));
	
	v += 0.*texLookup(pos+op(-delta, -delta, -delta));
	*/
	
	for(auto fac  : {-1., 1.})
	//for(unsigned char fac_mode=0; fac_mode <= 1; fac_mode++)
	  {
	    //double fac = -1.;
	    //if(fac_mode)
	    //fac = 1.;

	    
	    v +=
	      fac*
	      (
	       weights.x * texLookup(pos+op(-delta, -delta, -delta*fac))
	       + weights.y * texLookup(pos+op(-delta, 0, -delta*fac))
	       + weights.z * texLookup(pos+op(-delta, delta,-delta*fac))

	       + 2.*weights.x * texLookup(pos+op(0., -delta,-delta*fac))
	       + 2.*weights.y * texLookup(pos+op(0., 0,-delta*fac))
	       + 2.*weights.z * texLookup(pos+op(0., delta,-delta*fac))
      
	       + weights.x * texLookup(pos+op(delta, -delta,-delta*fac))
	       + weights.y * texLookup(pos+op(delta, 0,-delta*fac))
	       + weights.z * texLookup(pos+op(delta, delta,-delta*fac))
	       )
	      ;
	  }
	return v;
      };


    /*
    out.x = processDir(swapX);
    out.y = processDir(swapY);
    out.z = processDir(swapZ);
    */
    out.x = processDir(0);
    out.y = processDir(1);
    out.z = processDir(2);
        
    return out;    
  }

  

  template<typename V, typename POS, typename DIM>
    #ifdef USE_CUDA_RUN_DEVICE
  __host__ __device__
#endif
  POS sobelFeldman3D(const V& data, POS pos, DIM dim, const double delta)
  {
    return
      sobelFeldman3D(
		     [&data, &dim](POS pos)
		     {return texLookupLinear3D(pos, data, dim);},
		     pos, delta
		     );
    
#if 1
#else
    POS out(0., 0., 0.);

    const std::vector<double> weights = {3., 10., 3.};

    auto processDir = [&](auto op)
      {
	double v = 0.;
	for(auto fac  : {-1., 1.})
	  {
	    v +=
	      fac*
	      (
	       weights.x * texLookupLinear3D(pos+op(-delta, -delta, -delta*fac), data, dim)
	       + weights.y * texLookupLinear3D(pos+op(-delta, 0, -delta*fac), data, dim)
	       + weights.z * texLookupLinear3D(pos+op(-delta, delta,-delta*fac), data, dim)

	       + 2.*weights.x * texLookupLinear3D(pos+op(0., -delta,-delta*fac), data, dim)
	       + 2.*weights.y * texLookupLinear3D(pos+op(0., 0,-delta*fac), data, dim)
	       + 2.*weights.z * texLookupLinear3D(pos+op(0., delta,-delta*fac), data, dim)
      
	       + weights.x * texLookupLinear3D(pos+op(delta, -delta,-delta*fac), data, dim)
	       + weights.y * texLookupLinear3D(pos+op(delta, 0,-delta*fac), data, dim)
	       + weights.z * texLookupLinear3D(pos+op(delta, delta,-delta*fac), data, dim)
	       )
	      ;
	  }
	return v;
      };

    auto swapX = [](double x, double y, double z)
      {return POS(z, x, y);};
    
    auto swapY = [](double x, double y, double z)
      {return POS(x,z, y);};
    
    auto swapZ = [](double x, double y, double z)
      {return POS(x,y,z);};

    out.x = processDir(swapX);
    out.y = processDir(swapY);
    out.z = processDir(swapZ);
        
    return out;
    #endif
  }
}

#endif // __HELPER_GRADIENT__
