#ifndef __HELPER_TRIANGULATION__
#define __HELPER_TRIANGULATION__

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Triangulation_3.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <unordered_map>
#include <unordered_set>
#include "helper/helper_idx.h"
//#include "distHeuristics.h"
//#include "drawFlowGraph.h"

//#include "helper_Comp.h"

namespace helper_triangulation
{

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
                     
  typedef CGAL::Delaunay_triangulation_2<K> Triangulation_2;
  typedef CGAL::Delaunay_triangulation_3<K> Triangulation_3;
  typedef Triangulation_2::Point Point2;
  typedef Triangulation_3::Point Point3;

  /*
  template <template<class> class H, class S>
void f(const H<S> &value) {
}
  */

  template<typename T>
    void pointsFromEdgeIt(Point3& pa, Point3& pb, const T& it)
    {
      pa = it->first->vertex(it->second)->point();
      pb = it->first->vertex(it->third)->point();            
    }

  //http://stackoverflow.com/questions/9683809/cgal-2d-delaunay-triangulation-how-to-get-edges-as-vertex-id-pairs
  template<typename T>
    void pointsFromEdgeIt(Point2& pa, Point2& pb, const T& it)
    {
      //assert(it->second >= 0 && it->second <= 2);
      pa = it->first->vertex((it->second+1)%3)->point();
      pb = it->first->vertex((it->second+2)%3)->point();

      //std::cout << "P " << pa.x() << " " << pa.y() << " "
      //<< pb.x() << " " << pb.y() << std::endl;
    }

  int3 p2iii(const Point3& p)
  {
    return make_int3(p.x(), p.y(), p.z());
  }

  int3 p2iii(const Point2& p)
  {
    return make_int3(p.x(), p.y(), 0);
  }

  
  void iii2p(Point3& p, const int3& i3)
  {
    p = Point3(i3.x, i3.y, i3.z);
  }

  void iii2p(Point2& p, const int3& i3)
  {
    assert(i3.z==0);
    p = Point2(i3.x, i3.y);
  }


  template<typename FV>
    void to_p(Point3&p, const FV& f)
  {
    p = Point3(f.x, f.y, f.z);    
  }

  template<typename FV>
    void to_p(FV& f, const Point3& p)
  {
    f.x = p.x();
    f.y = p.y();
    f.z = p.z();
  }

  template<typename FV>
    void to_p(FV& f, const Point2& p)
  {
    f.x = p.x();
    f.y = p.y();
  }

  template<typename FV>
    void to_p(Point2&p, const FV& f)
  {
    p = Point2(f.x, f.y);
  }

  
  template<typename Tri_t, typename Point, typename EDGES, typename P_it>
    void
    delaunayEdgesT(EDGES& edges,
                   //const std::unordered_map<UI, I>& supply,
                   const P_it& pos_begin, const P_it& pos_end/*,
                                                               const std::string graphFName=""*/)
  {

    std::list<Point> L;

    struct Comp
    {
      bool operator()(const Point3& a, const Point3& b) const
      {
        return a.x()<b.x() || (a.x()==b.x () && (a.y()<b.y() || (a.y()==b.y() && a.z() < b.z())));
      }

      bool operator()(const Point2& a, const Point2& b) const
      {
        return a.x()<b.x() || (a.x()==b.x() && (a.y()<b.y()));
      }
    };

    typedef typename P_it::value_type P;
    //typedef typename PV::value_type P;
    std::map<Point, uint32_t, Comp> pos2idx;
    //pos2idx.reserve(supply.size());
	  
                         
    //Std::cout << "there are " << supply.size() << " vertices\n";
    size_t cnt = 0;
    {
      
      //for(const auto& pm: pos)
      for(P_it it = pos_begin; it != pos_end; it++)
          {
            Point p;
            to_p(p, *it);
            
            const bool success
              = pos2idx.insert(std::make_pair(p, cnt)).second;
            cnt++;
            assert(success);
            
            L.push_back(p);

            /*
            {
              P a = *it;
              P b;
              to_p(b, p);
              std::cout << "a " << a.x << " " << a.y << " " << a.z
                        << " b " << b.x << " " << b.y << " " << b.z
                        << std::endl;              
              assert(!helper_Comp()(a,b) && !helper_Comp()(b,a));
              assert(pos2idx.find(b) != pos2idx.end());
            }
            */
          }
    }
          
    //std::cout << "there are " << L.size() << " points\n";
    Tri_t T(L.begin(), L.end());
    assert( T.is_valid() ); // checking validity of T

    //assert(L.size() == pos.size());
    //std::cout << T.number_of_vertices() << " vs " << supply.size() << std::endl;
    //assert(T.number_of_vertices() == pos.size());

    edges.reserve(16*cnt);
    for(auto it = T.finite_edges_begin(); it != T.finite_edges_end(); it++)
      {
	      
        Point pa, pb;
        pointsFromEdgeIt(pa, pb, it);
        
        const auto ita = pos2idx.find(pa);
        assert(ita != pos2idx.end());

        const auto itb = pos2idx.find(pb);
        assert(itb != pos2idx.end());

        const uint32_t i0 = std::min(ita->second, itb->second);
        const uint32_t i1 = std::max(ita->second, itb->second);

        edges.emplace_back(i0, i1);
        //if(std::find(edges[i0].begin(), edges[i0].end(), i1) != edges[i0].end())
        //edges[i0].push_back(i1);
      }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end() ), edges.end() );
          
    /*
      if(graphFName != "")
      {
      if(supply.size() < 100000)
      drawFlowGraph(graphFName, supply, pos2idx, costEdges, volDim);
      else
      std::cout << "do not draw graph, it has too many nodes\n";
      }
    */
	
  }


  template<typename EDGES, typename P_it>
    void
    delaunayEdges(EDGES& edges,
                  const P_it pos_begin,
                  const P_it pos_end,
		  int nD,
		  const std::string graphFName="")
  {
    if(nD == 3)
      delaunayEdgesT
	<Triangulation_3,
	Point3>
	(edges, pos_begin, pos_end);
    else if(nD == 2)
      delaunayEdgesT
	<Triangulation_2,
	Point2>
	(edges, pos_begin, pos_end);
    else
      assert(false);
  }

#if 0
  template<typename I, typename UI>
    std::vector<std::tuple<UI, UI, I>>
    delaunayWeightedEdges(const std::unordered_map<UI, I>& supply,
                          int3 volDim, double edgeCostFac,
			  const std::string graphFName="")
    {
      
      std::vector<std::tuple<UI, UI, I>> costEdges;

      delaunayEdges(costEdges, supply,
		    volDim, graphFName);

      std::vector<I> idx2pos;
      idx2pos.resize(supply.size());
      {
	size_t idx = 0;
	for(auto e : supply)
	  {
	    idx2pos[idx] = std::get<0>(e);
	    idx++;
	  }
      }
	
      const size_t nEdges = costEdges.size();
      costEdges.resize(nEdges*2);
      for(size_t i=0; i<nEdges; i++)
	{
	  const auto ida = std::get<0>(costEdges[i]);
	  const auto idb = std::get<1>(costEdges[i]);

	  assert(ida < idx2pos.size());
	  assert(idb < idx2pos.size());

	  const auto iiida = i2iii(idx2pos[ida], volDim);
	  const auto iiidb = i2iii(idx2pos[idb], volDim);
	  
	  //const auto iiida = i2iii(ida, volDim);
	  //const auto iiidb = i2iii(idb, volDim);
	      
	  const I cost =
	    edgeCostFac*distMetricHeuristic(iiida-iiidb);

	  std::get<2>(costEdges[i]) = cost;
	  costEdges[i+nEdges] = std::make_tuple(idb, ida, cost);
	}	  	  	  	  
	
      return costEdges;
    }

  template<typename I, typename UI>
    std::vector<std::tuple<UI, UI, I>>
    fullWeightedEdges(const std::unordered_map<UI, I>& supply,
                          int3 volDim, double edgeCostFac,
			  const std::string graphFName="")
    {
      //assert(supply.size() < 300000);
      const size_t nEdgesEstimate = supply.size()*supply.size();
      {
        const size_t nBytes_GB =
#ifndef USE_OMP_EVAL_SELECTION
	  14*
#endif
	  1024ul*1024ul*1024ul;
        const size_t nBytesRequired = nEdgesEstimate*sizeof(std::tuple<UI, UI, I>);
        const size_t nBytesMax = nBytes_GB;
        if(nBytesRequired > nBytesMax)
          {
            std::stringstream ss;
            ss << __func__
               << " required " << nBytesRequired/static_cast<double>(nBytes_GB) << "GB"
               << " max " << nBytesRequired/static_cast<double>(nBytes_GB) << "GB"
               << std::endl;
            std::cerr << ss.str();
            std::cout << ss.str();
            exit(-1);
          }
      }
      std::vector<std::tuple<UI, UI, I>> costEdges;
      costEdges.reserve(nEdgesEstimate);

      std::vector<I> idx2pos;
      idx2pos.resize(supply.size());
      
      {
	size_t eidx = 0;
	for(auto e: supply)
	  {
	    idx2pos[eidx] = std::get<0>(e);
	    
	    if(e.second > 0)
	      {
		size_t fidx = 0;
		for(auto f: supply)
		  {
		    if(f.second < 0)
		      costEdges.emplace_back(eidx, fidx, 0);
		    fidx++;
		  }
	      }
	    eidx++;
	  }
      }

      
      
            
      const size_t nEdges = costEdges.size();      
      for(size_t i=0; i<nEdges; i++)
	{
	  const auto ida = std::get<0>(costEdges[i]);
	  const auto idb = std::get<1>(costEdges[i]);

	  assert(ida < idx2pos.size());
	  assert(idb < idx2pos.size());

	  const auto iiida = i2iii(idx2pos[ida], volDim);
	  const auto iiidb = i2iii(idx2pos[idb], volDim);
	      
	  const I cost =
	    edgeCostFac*distMetricHeuristic(iiida-iiidb);

	  std::get<2>(costEdges[i]) = cost;	  
	}	  	  	  	  
	
      return costEdges;
    }
#endif
};

#endif
