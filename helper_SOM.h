#ifndef __HELPER_SOM__
#define __HELPER_SOM__

#ifndef M_VEC
#define M_VEC
#endif
#include <array>
#include <random>
#include "helper_util.h"
#include <tuple>
#ifndef NO_OMP
#include <omp.h>
#endif
#include <sstream>
#include "helper_asciiFile.h"
#include "helper_string.h"

namespace helper
{
  
  template<std::size_t _nUnits, typename F>
    std::array<F,  _nUnits> operator-(const std::array<F,  _nUnits>& lhs, const std::array<F,  _nUnits>& rhs)
    {
      std::array<F,  _nUnits> out;

      for(uint64_t i=0; i<lhs.size(); i++)
	out[i] =lhs[i]-rhs[i];
      
      return out;
    }

  template<std::size_t _nUnits, typename F>
    std::array<F,  _nUnits> operator+(const std::array<F,  _nUnits>& lhs, const std::array<F,  _nUnits>& rhs)
    {
      std::array<F,  _nUnits> out;

      for(uint64_t i=0; i<lhs.size(); i++)
	out[i] =lhs[i]+rhs[i];
      
      return out;
    }

  template<std::size_t _nUnits, typename F>
  void operator+=(std::array<F,  _nUnits>& lhs, const std::array<F,  _nUnits>& rhs)
    {

      for(uint64_t i=0; i<lhs.size(); i++)
	lhs[i]+=rhs[i];           
    }

  template<std::size_t _nUnits, typename F>
    std::array<F,  _nUnits> operator*(const std::array<F,  _nUnits>& lhs, const F& rhs)
    {
      std::array<F,  _nUnits> out;

      for(uint64_t i=0; i<lhs.size(); i++)
	out[i] =lhs[i]*rhs;
      
      return out;
    }

  template<std::size_t _nUnits, typename F>
  std::array<F,  _nUnits> operator*(const F& lhs, const std::array<F,  _nUnits>& rhs)
    {
      return rhs*lhs;
    }

  template<std::size_t _nDims, typename F>
inline std::ostream& operator<< (std::ostream &out, const  std::array<F,  _nDims> v)
{

  out << "(";
  for(auto& e : v)
    out << " " << e;
  out << " )";    

    
  return out;
}
    

  template<bool _isRing, std::size_t _nDims, std::size_t _nUnits, typename DIST, typename F>
  class helper_SOM_norm_ring
  {
  public:
    using vec_t = std::array<F,  _nDims>;


    // F length2(const vec_t& v) const
    // {
    //   F l=0;
    //   for(const auto e : v)
    // 	l += e*e;
    //   return l;
    // }
    
    // void operator-=(Vec &a, const Vec b)
    // {
    //   a.x -= b.x;
    //   a.y -= b.y;
    // }

    class TrainingInstructor
    {
    public:

      TrainingInstructor()
      {
	init();
      }
      
      void init()
      {
	_radius = 0.5;
	_rate = 0.05;
      }

      int64_t neighborhood() const
      {
	return _radius*_nUnits+.5;
      }

      F rate(int64_t i) const
      {	
	const auto n = neighborhood();

	assert(i<=n);
	return _rate*static_cast<F>(n-std::abs(i)+1)/(n+1);	
      }

      void moveOn()
      {
	const F fac = 0.98;
	_radius *=fac;
	_rate *= fac;
      }
      
      F _radius;
      F _rate;      
    };
  
    helper_SOM_norm_ring(DIST dist) :
      _dist(dist)
    {
      init();
    }

    template<typename T>
    helper_SOM_norm_ring(DIST dist, const T& vecs) :
      _dist(dist)
    {
      init(vecs);
    }

    template<typename T>
    void init(const T& vecs)
    {
      //std::copy(vinit.begin(), vinit.end(), _d.begin());
      std::uniform_int_distribution<> dis(0, vecs.size()-1);
      
      for(auto& v : _d)
	//for(auto& e : v)
	for(size_t i=0; i<v.size(); i++)
	  v[i] = vecs[dis(_rng)][i];      
    }
    
    void init()
    {
      for(auto& v : _d)
	for(auto& e : v)
	  e = _dis(_rng);
    }

    template<typename T_it>
    auto bestMatchingUnits(T_it vin_begin, T_it vin_end) const
    {
      const size_t n = vin_end - vin_begin;
      std::vector<std::tuple<int64_t, F>> bmus(n);
#pragma omp parallel for
      for(size_t i=0;i<n; i++)
	bmus[i] = bestMatchingUnit(vin_begin[i]);
      
      return bmus;
    }

    template<typename T_it>
    auto representativeElemPerUnit(T_it vin_begin, T_it vin_end) const
    {
      const size_t n = vin_end - vin_begin;
      std::array<std::tuple<int64_t, F>, _nUnits> units;
      std::fill(units.begin(), units.end(), std::tuple<int64_t, F>(-1, std::numeric_limits<F>::max()));

      const auto maxNThreads =
#ifndef NO_OMP
	omp_get_max_threads()
#else
	1
#endif
	;
      std::vector<std::tuple<int64_t, F>> rslts(maxNThreads);
      
      for(size_t u=0; u<_nUnits; u++)
	{
	  const auto unit = _d[u];
	  
	  std::fill(rslts.begin(), rslts.end(), std::tuple<int64_t, F>(-1, std::numeric_limits<F>::max()));

#ifndef NO_OMP
#pragma omp parallel for
#endif
	  for(size_t i=0; i<n; i++)
	    {
	      const auto threadId = 
#ifndef NO_OMP
	      omp_get_thread_num();
#else
	      0;
#endif
	      const auto d = _dist(vin_begin[i], unit);
	      if(d < std::get<1>(rslts[threadId]))
		{
		  rslts[threadId] = std::tuple<int64_t, F>(i, d);
		}
	    }

	  for(const auto e : rslts)
	    if(std::get<1>(e) < std::get<1>(units[u]))
	      units[u] = e;
	}
      return units;
    }
    
    std::tuple<int64_t, F> bestMatchingUnit(const vec_t& vin) const
    {
      F smallest_d = std::numeric_limits<F>::max();
      int64_t smallest_i = 0;
      
      for(int64_t i=0; i<_d.size(); i++)
	{
	  const auto unit = _d[i];
	  const auto d =
	    //length2(vin- _d[i]);
	    _dist(vin, unit);
	  
	  if(d<smallest_d)
	    {
	      smallest_d = d;
	      smallest_i = i;
	    }
	}

      return std::make_tuple(smallest_i, smallest_d);
    }

    template<typename T_it>
    F train(const T_it vin_begin, const T_it vin_end)
    {     
      //
      // find best matching unit
      //

      const auto nVecs = vin_end - vin_begin;
      const auto bmus = bestMatchingUnits(vin_begin, vin_end);

      assert(bmus.size() == nVecs);
      F value = 0;      
      
      for(size_t i=0; i<nVecs; i++)
	{
	  //std::cout << "ahhhhhhhhhhh " << i << std::endl;
	  value += std::get<1>(bmus[i]);
	}

#pragma omp parallel for
      for(size_t j=0; j<nVecs; j++)
	{
	  const int64_t smallest_i = std::get<0>(bmus[j]);
	  const auto vin = vin_begin[j];
      
	  //
	  // adjust vectors
	  //
	  const int64_t neighborhoodRadius = _instructor.neighborhood();

	  //#pragma omp parallel for
	  for(int64_t i=smallest_i-neighborhoodRadius;
	      i<=smallest_i+neighborhoodRadius; i++)
	    {
	      const auto r = _instructor.rate(smallest_i - i);

	      assert(_d.size() == _nUnits);

	      int64_t idx;

	      if(_isRing)
		{
		  idx = (i+_nUnits) % _nUnits;
		}
	      else
		{
		  if(i < 0 || i >= _nUnits)
		    continue;
		  idx = i;
		}

	      assert(idx >= 0 && idx < _nUnits);

	      
	      // std::stringstream ss;
	      // ss<< "prev: ";
	      // ss << _d[idx];
	      // ss << std::endl;
	      // ss<< "match: " << vin << std::endl;
	      
	      _d[idx] +=  r* (vin-_d[idx]);

	      for(auto & e: _d[idx])
		e= std::max(e, 0.);
	      // ss << "new: " << _d[idx] << std::endl;

	      // std::cout << ss.str();
	      
	    }
	}
      return value;
    }
      
    /*
    auto train(const F* in)
    {
       vec_t vin;
       std::copy(in, in+vin.size(), vin.begin());
       return train(vin);
    }
    */

    void moveOn()
    {
      _instructor.moveOn();
    }

    auto getUnitDists_sequence() const
    {
      std::array<F,  _nUnits> dists;

      const auto n = _isRing ? dists.size() : (dists.size()-1);

      if(!_isRing)
	dists[_nUnits-1] = 0.;
#pragma omp parallel for
      for(size_t i=0; i<n; i++)
	dists[i] =
	  //std::sqrt(length2(_d[i]-_d[(i+1)%_nUnits]));
	  _dist(_d[i], _d[(i+1)%_nUnits]);
      return dists;
    }

    auto getUnitDists_matrix() const
    {
      std::array<F,  _nUnits*_nUnits> dists;
      std::fill(dists.begin(), dists.end(), 0);
      
#pragma omp parallel for
      for(size_t i=0; i<_nUnits; i++)
	for(size_t j=i+1; j<_nUnits; j++)
	  {
	    const auto d = _dist(_d[i], _d[j]);
	    dists[i+j*_nUnits] = d;
	    dists[j+i*_nUnits] = d;
	    //std::sqrt(length2(_d[i]-_d[(i+1)%_nUnits]));	      
	  }
      return dists;
    }

    bool storeUnits(const std::string fname) const
    {
      std::vector<std::string> v;

      for(const auto unit : _d)
	{
	  std::stringstream ss;
	  for(size_t i=0; i<unit.size(); i++)
	    {
	      ss << unit[i];
	      if(i < unit.size()-1)
		ss << " ";
	    }
	  v.push_back(ss.str());
	}      
      return helper::writeASCIIv(v, fname);
    }

    bool loadUnits(const std::string fname)
    {
      std::vector<std::string> v;
      if(!helper::readASCIIv(v, fname))
	return false;

      assert(v.size() == _nUnits);

      for(size_t i=0; i<_nUnits; i++)
	{
	  const auto tokens = helper::split(v[i], ' ');
	  assert(tokens.size() == _nDims);
	  for(size_t j=0; j<_nDims; j++)
	    _d[i][j] = std::stod(tokens[j]);
	}

      return true;
    }

    auto getUnits() const
    {
      return _d;
    }
    
    std::array<vec_t,  _nUnits> _d;

    std::mt19937 _rng = std::mt19937(23);
    std::uniform_real_distribution<F> _dis  = std::uniform_real_distribution<F>(0., 1.);

    TrainingInstructor _instructor;

    const DIST _dist;
  };


  template<bool _isRing, std::size_t _nDims, std::size_t _nUnits, typename DIST, typename F>
  inline std::ostream& operator<< (std::ostream &out, const helper_SOM_norm_ring<_isRing, _nDims, _nUnits, DIST, F> &som)
  {
    out << "[";
    for(auto& v : som._d)
      {
	out << v;
	// out << "(";
	// for(auto& e : v)
	//   out << " " << e;
	// out << " )";
      }

    out << "]";

    out << "rate: " << som._instructor._rate << "; radius: " << som._instructor._radius;
    return out;
  }

}
#endif //__HELPER_SOM__
