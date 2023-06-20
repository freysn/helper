#ifndef __SELECT__BONE__
#define __SELECT__BONE__

/*
  selection of k optimal visualization designs
  based on: flotis_fysn/scripts/select_dynProg.h
  first and last time step are always chosen
  determine selection (i.e., path) with maximum cost
 */

#include <iostream>
#include <vector>
#include <map>
#include <list>
#include "helper/helper_util.h"
#include "helper/helper_idx.h"

/*
  
 */
std::pair<std::list<uint32_t>, double>
select_dynProg_max(
	       const size_t k, // the number of time steps to select
	       const std::vector<double>& dists,
	       const size_t nTimeSteps)
{
  const bool verbose=false;
  assert(dists.size()==nTimeSteps*nTimeSteps);


  using nSel_timeStep_map_t = std::map<std::pair<size_t, size_t>, std::pair<double, std::list<uint32_t>>>;
  nSel_timeStep_map_t  nSel_timeStep_map;

  auto d = [&dists, nTimeSteps](size_t b0, size_t b1)
  {
    assert(b0 <= b1);

    const auto v=dists[b1+nTimeSteps*b0];
    hassertm2(v != std::numeric_limits<double>::lowest(), b0,b1);
    //return v*v;
    return v;
  };

  auto update = [&nSel_timeStep_map,d](const auto nSel, const auto t, const auto e)
  {

    auto l = e.second;

    const auto last_t = l.back();    
    l.push_back(t);

    auto x = e.first;


    
    //std::cout << "evaluate dist " << last_t << " " << t << ": " << d(last_t, t) << std::endl;
    x += d(last_t, t);
      /*  
    //
    // TODO: CHANGE BELOW FOR NEW FORMAT
    // distances are already pre-aggregated there
    //
    //
    // this accumulates the value for the new selection
    //
    for(const auto u : helper::range_be(last_t+1, t))
      {
	//
	// in d() the first parameter denotes the selected candidate, the second one refers to the beta value
	//
	x+=std::min(d(last_t, u), d(t,u));
      }
      */

      
    const auto key=std::make_pair(nSel,t);
    const auto value=std::make_pair(x,l);
    auto it=nSel_timeStep_map.find(key);
    
    //bool newEntry=true;
    if(it==nSel_timeStep_map.end())
      {
	nSel_timeStep_map[key]=value;
      }
    // update if new value is larger
    else if(it->second.first < value.first)
      {
	it->second=value;
	//newEntry=false;
      }
  };

  /*
  // first selection is always first time step
  for(const auto t : helper::range_be(1,nTimeSteps-k))
  //update(1, t, d(0, t), std::list<uint32_t>(1,0));
  update(1, t, std::make_pair(0., std::list<uint32_t>(1,0)));
  */
  
  nSel_timeStep_map[std::make_pair(1,0)]=std::make_pair(0., std::list<uint32_t>(1,0));


  
  for(const auto nSel : helper::range_bn(2, k-1))
    {
      if(verbose)
	{
	  std::cout << "---------------- nSel " << nSel << std::endl;
	  for(const auto & e : nSel_timeStep_map)
	    {	  
	      std::cout << e.first.first << " " <<  e.first.second << " | " << e.second.first << "[";
	      for(const auto & f : e.second.second)
		std::cout << " " << f;
	      std::cout << "]"<<std::endl;
	    }
	}
      
      nSel_timeStep_map_t nSel_timeStep_map_prev;
      std::swap(nSel_timeStep_map_prev, nSel_timeStep_map);
      
      for(const auto & e : nSel_timeStep_map_prev)
	{	  
	  assert(e.first.first==nSel-1);

	  if(nSel<=k-1)
	    {
	      for(const auto t :
		    helper::range_be(e.first.second+1,
				     nTimeSteps-(k-nSel)))
		{
		  //auto tmpl=e.second.second;
		  //tmpl.push_back(t);
		  //update(nSel, t, e.second.first+d(e.first.second, t), tmpl);

		  /*
		  std::cout << "update(" << "nSel=" << nSel << ", t=" << t << "e.second.first=" << e.second.first<< ", e.second.second=[";
		  for(const auto f : e.second.second)
		    std::cout << " " << f;
		  std::cout << " ]"<< std::endl;
		  */
		  update(nSel, t, e.second);
		}
	    }
	  // the last time step is always selected
	  else //nSel==k
	    {
	      update(nSel, nTimeSteps-1, e.second);
	    }
	}
    }

  if(verbose)
    std::cout << "final key: " << k << " " <<  nTimeSteps-1 << std::endl;
  const auto & rslt = nSel_timeStep_map[std::make_pair(k, nTimeSteps-1)];

  if(verbose)
    {
      std::cout << "final cost: " << rslt.first/*/nTimeSteps*/
		<< std::endl;

      for(const auto e : rslt.second)
	std::cout << e << " ";

      std::cout << std::endl;
    }
  //return std::make_pair(rslt.second, rslt.first/nTimeSteps);
  return std::make_pair(rslt.second, rslt.first);
}


#endif // __SELECT__DYN_PROG__
