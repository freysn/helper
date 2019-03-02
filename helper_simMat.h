#ifndef __HELPER_SIM_MAT__
#define __HELPER_SIM_MAT__


#include <vector>
#include <string>
#include <limits>
#include "helper_cimg.h"
#include <tuple>
#include <fstream>
#include <list>
#include "volData/splitStr.h"

namespace helper
{

  template<bool IDENTITY=false>
class IndicesSM
{
public:

 IndicesSM(size_t _nT=0) : nT(_nT)
    {
      t1 =  !IDENTITY;
    }

static bool valid(size_t t0, size_t t1, size_t nT)
{
  return (t0 < t1 +IDENTITY) && t1 < nT;
}

bool valid()
{
  return valid(t0, t1, nT);
}

IndicesSM& operator++()
  {
    t0++;
    if(!(t0<t1+IDENTITY))
      //      return true;
      //    else
      {
	t0=0;
	t1++;
	//return (t1<nT);
      }
    return *this;
  }

size_t get() const
{
  return t1+t0*nT;
}
  
  size_t nT = 0;
  size_t t0=0;
  size_t t1=0;
};

  template<bool IDENTITY=false>
  size_t nEntriesSM(size_t nT)
  {
    return ((nT-1)*nT)/2 + IDENTITY * nT;
  }
  
template<bool IDENTITY=false>
  std::vector<std::pair<size_t, size_t>> id2tMap_denseSM(size_t nT)
    {
      std::vector<std::pair<size_t, size_t>> out;
      out.reserve(nEntriesSM<IDENTITY>(nT));
      IndicesSM<> indices(nT);
      while(indices.valid())
	{
	  out.emplace_back(indices.t0, indices.t1);
	  ++indices;
	}
      assert(nEntriesSM<IDENTITY>(nT) == out.size());
      return out;
    }
  
  double maxValueSM(const std::vector<double>& simMat)
  {
    double maxV = 0.;
    for(size_t i=0; i<simMat.size(); i++)
      {
	if(simMat[i] != std::numeric_limits<double>::max())
	  maxV = std::max(maxV, simMat[i]);
      }
    return maxV;
  }

  void clearDuplicatesSM(std::vector<double>& simMat, size_t nT, bool clearIdentity=true)
  {
    for(size_t t1=0; t1<nT; t1++)
      for(size_t t0=0; t0<=t1; t0++)
	{
	  if(!clearIdentity && (t0==t1))
	    continue;
	  // this is the reverse of the standard addressing scheme
	  simMat[t0 + t1*nT] = std::numeric_limits<double>::max();
	}
  }
  
  void outputSM(const std::vector<double>& simMat, size_t nT, std::string outPath)
  {
  
    //const double maxV = *max_element(std::begin(simMat_255), std::end(simMat_255));

    const double maxV = maxValueSM(simMat);

    std::vector<double> simMat_255(4*simMat.size(), 0.);
    for(size_t i=0; i<simMat.size(); i++)
      {
	double v = simMat[i];
	if(v==std::numeric_limits<double>::max())
	  {
	    simMat_255[i*4+0] = 255.;
	  }
	else
	  {
	    v = (v/maxV)*255.;
	    simMat_255[i*4+0] = v;
	    simMat_255[i*4+1] = v;
	    simMat_255[i*4+2] = v;
	  }

	simMat_255[i*4+3] = 255.;
      }
    helper::cimgWrite(outPath+"simMat.png", simMat_255, make_V3<int>(nT, nT, 1), 4);
  }

  std::tuple<std::vector<double>, size_t>
    readSM(std::string fname)
    {
      
      std::ifstream infile(fname);
      if(!infile.is_open())
	return std::make_tuple(std::vector<double>(), 0);
    
      std::list<std::tuple<size_t, size_t, double>> distsList;

      size_t n=0;
      while (! infile.eof() )
	{
	  std::string line;
	  std::getline(infile,line);
	  if(line=="")
	    continue;
        
	  std::vector<std::string> strs = split(line, ' ');
	  //std::cout << "line: |" << line << "|\n";
	  assert(strs.size()==3);
	  size_t a = std::stoi(strs[0]);
	  size_t b = std::stoi(strs[1]);
	  double d = std::stof(strs[2]);
	  distsList.emplace_back(a,b,d);
	  n = std::max(std::max(a,b)+1, n);                
	}

      //std::numeric_limits<double>::max()
      std::vector<double> dists(n*n, 0.);
      std::vector<size_t> cnt(n*n,0);
      for(auto e : distsList)
	{
	  const size_t i0 = std::max(std::get<0>(e),std::get<1>(e));
	  const size_t i1 = std::min(std::get<0>(e),std::get<1>(e));

	  const size_t idx0 = i0+ n*i1;
	  dists[idx0] += std::get<2>(e);
	  cnt[idx0]++;
	  //dists[i1+ n*i0] += std::get<2>(e);        
	}

      for(size_t i1=0; i1<n; i1++)
	for(size_t i0=i1; i0<n; i0++)
	  {
	    const size_t idx0 = i0+ n*i1;
	    if(cnt[idx0] == 0)
	      dists[idx0] = std::numeric_limits<double>::max();
	    else
	      dists[idx0] /= cnt[idx0];
	    dists[i1+n*i0] = dists[idx0];
	  }
    
      return std::make_tuple(dists,n);
    }

  std::tuple<std::vector<double>, size_t>
    genSin(const size_t nT = 400)
    {
      //#pragma message(TODO "something")

      std::cout << "compute similarity matrix with " << nT << " time steps\n";
  
      std::vector<double> simMat(nT*nT);
      {
	auto func = [](double v)
	  {
	    if(false)
	      return sin(std::sqrt(0.01*v+1.)*(v/40.));
	    else
	      return sin(v/4.);
	  };
    
	for(size_t t1=0; t1<nT; t1++)
	  for(size_t t0=0; t0<nT; t0++)
	    simMat[t0+t1*nT] = std::abs(func(t0) - func(t1));    
      }
      return std::make_tuple(simMat, nT);
    }

}

#endif // __HELPER_SIM_MAT__
