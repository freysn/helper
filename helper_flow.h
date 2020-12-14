#ifndef __FLOW_HELPER__
#define __FLOW_HELPER__

#ifdef USE_OR_TOOLS

// Copyright 2010-2014 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


//#include "base/commandlineflags.h"
//#include "base/logging.h"
#include <cstdlib>
//#include <helper_math.h>
#include "graph/ebert_graph.h"
#include "graph/min_cost_flow.h"
#include "graph/connectivity.h"
#include "../volData/VolDataHandlerCUDA.h"
#include "../volData/volData.h"
//#include "flowShortcuts.h"
#include <tuple>
#include <algorithm>

#include <set>
#include <list>
#include <random>
#include <chrono>
#include <typeindex>
#include "volData/equalizeSum.h"
//#include "helper_idx.h"

template<typename Q, typename D=double>
class fillMassOpts_t
{
  /*
public:
  fillMassOpts_t()
  {
    //maxIdxOffset = std::numeric_limits<size_t>::max();
    //offsetMass = 1;
    //offsetSum = 0;
    //massMultiply = 1;
    generatorCostFac = 1.;

    singleMassGenerator = false;
  };

D generatorCost(const int3& volDim) const
{
  D cost = volDim.x+volDim.y+volDim.z;
  return cost*generatorCostFac; 
}
//size_t maxIdxOffset;
//Q offsetMass;
//Q offsetSum;
//Q massMultiply;
D generatorCostFac;
bool singleMassGenerator;
  */
};

#if 0
template<typename Q>
std::tuple<std::vector<Q>, Q> fill2equalizeMass(std::vector<Q>& supply_nodes, fillMassOpts_t<Q> fillMassOpts)
{
#ifdef VERBOSE
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
#endif
  
  std::vector<Q> flowCreator(supply_nodes.size(), 0);
  typedef std::mt19937 MyRNG;
  uint32_t seed_val=123;
  
  MyRNG rng;
  rng.seed(seed_val);
  
  
  Q sumNeg = 0;
  Q sumPos = 0;
  for(size_t i=0; i<supply_nodes.size(); i++)
    {
      //const Q d = std::get<1>(vfrom[i])-std::get<1>(vto[i]);
      const Q d = supply_nodes[i];
      
      if(d<0)
        sumNeg -= d;
      else
        sumPos += d;
    }

  const Q sum = sumPos-sumNeg + fillMassOpts.offsetSum;
      
  if(sum == 0)
    return std::make_tuple(flowCreator, 0);

  //Q sumDiffOld = sum;
  //
  // the sum of input and output does not match,
  // i.e., mass needs to be generated or destroyed
  // we do this, by virtually adding mass to the component
  // where there is not enough.
  // sum < 0:
  //  [input mass] > [output mass]
  //  add mass to output
  //  flow needs to be adjusted toward end
  // sum > 0:
  //  vice versa
  const Q deltaAbs = 1;

  
  Q delta = deltaAbs;
  if(sum>0)
    delta=-deltaAbs;

  
  Q sumDiff = sumPos;
          
  if(sum>0)
    sumDiff = sumNeg;

  sumDiff *= fillMassOpts.massMultiply;
  sumDiff += std::min(supply_nodes.size(), fillMassOpts.maxIdxOffset)*fillMassOpts.offsetMass;
  
  assert(sumDiff > 0);
  std::uniform_int_distribution<uint32_t>
    uint_dist(1,sumDiff);

  {
    std::vector<uint32_t> rngv(-sum/delta);
    for(auto& r : rngv)
      r = uint_dist(rng);
    std::sort(rngv.begin(), rngv.end());

    size_t idx=0;
    Q s=0;
    for(auto r : rngv)
      {
        while(r>s)
          {
            assert(idx < supply_nodes.size());
            if((supply_nodes[idx] < 0) && (sum > 0))
              s-=supply_nodes[idx]*fillMassOpts.massMultiply;
            else if((supply_nodes[idx] > 0) && (sum < 0))
              s+=supply_nodes[idx]*fillMassOpts.massMultiply;
              
            if(idx < fillMassOpts.maxIdxOffset)
              s+=fillMassOpts.offsetMass;
              
            idx++;
          }
        assert(idx>0);
        assert(idx<=supply_nodes.size());
        supply_nodes[idx-1] += delta;
        flowCreator[idx-1] += delta;
      }    
  }

#ifdef VERBOSE
  end = std::chrono::system_clock::now();
  std::cout <<  __func__ << "took " << std::chrono::duration_cast<std::chrono::milliseconds>
    (end-start).count() << "ms" << std::endl;
#endif

  return std::make_tuple(flowCreator, /*sumDiffOld*/sum);
}
#endif

  template<typename E, typename I>
  void createEdgesWithGenerator(E& costEdges, size_t generatorId, I costEdge2Generator, bool toGenerator, bool fromGenerator)
  {
    for(size_t i=0; i<generatorId; i++)
      {
        if(toGenerator)
          costEdges.emplace_back(i, generatorId, costEdge2Generator);
        if(fromGenerator)
          costEdges.emplace_back(generatorId, i, costEdge2Generator);      
      }
  }

template<typename Q>
size_t addGeneratorNode(std::vector<Q>& supply_nodes)
{
  const size_t generatorId = supply_nodes.size();
  supply_nodes.push_back(0);

  for(size_t i=0; i<generatorId; i++)
    supply_nodes[generatorId] -= supply_nodes[i];

  return generatorId;
}

template<typename Q, typename E>
void addGenerator(std::vector<Q>& supply_nodes,
                  E& cost_edges,
                  const Q& costEdge2Generator,
                  const bool bidirectional)
{
  const size_t generatorId = addGeneratorNode(supply_nodes);

  const bool toGenerator =
    bidirectional || (supply_nodes[generatorId] < 0);

  const bool fromGenerator =
    bidirectional || (supply_nodes[generatorId] > 0);
  
  createEdgesWithGenerator(cost_edges, generatorId, costEdge2Generator,
                           toGenerator, fromGenerator);
}

  // Test on a 4x4 matrix. Example taken from
// http://www.ee.oulu.fi/~mpa/matreng/eem1_2-1.htm
template<typename Q, typename E>
  std::tuple<Q, std::vector<Q>> mMinCostFlow(std::vector<Q> supply_nodes,
                                             /*std::list<std::tuple<I, I, Q>>*/E cost_edges) {
    
    using namespace operations_research;
    StarGraph graph(supply_nodes.size(), cost_edges.size());
    MinCostFlow min_cost_flow(&graph);

  std::list<ArcIndex> arcIndices;
  for(auto e : cost_edges)
    {
      ArcIndex arc = graph.AddArc(std::get<0>(e), std::get<1>(e));
      arcIndices.push_back(arc);
      min_cost_flow.SetArcUnitCost(arc, std::get<2>(e));
      //min_cost_flow.SetArcCapacity(arc, std::numeric_limits<Q>::max());
      min_cost_flow.SetArcCapacity(arc, std::numeric_limits<int>::max());
    }
  
  
  for (NodeIndex n=0; n<supply_nodes.size(); n++) 
    {
      min_cost_flow.SetNodeSupply(n, supply_nodes[n]);
    }

  CHECK(min_cost_flow.Solve());
  CHECK_EQ(MinCostFlow::OPTIMAL, min_cost_flow.status());
  CostValue total_flow_cost = min_cost_flow.GetOptimalCost();
  //std::cout << "total flow cost: " << total_flow_cost << std::endl;
  //CHECK_EQ(kExpectedCost, total_flow_cost);

  std::vector<Q> flow(arcIndices.size());
  {
    size_t cnt=0;
    for(auto a: arcIndices)
      {
        flow[cnt] = min_cost_flow.Flow(a);
        //std::cout << a << " " << min_cost_flow.Flow(a) << std::endl;
        cnt++;
      }
  }
  return std::make_tuple(total_flow_cost, flow);
}



  template<bool bipartite, typename I, typename I3>
  std::vector<std::tuple<I, I, I>>
  createFlowGraphEdges(const I3& volDim)
  {
    std::vector<std::tuple<I, I, I>> cost_edges;

  
    /*
      supply_nodes.push_back(1);
      supply_nodes.push_back(0);
      supply_nodes.push_back(-1);
      cost_edges.push_back(std::make_tuple(0,1,1));
      cost_edges.push_back(std::make_tuple(1,2,1));
      cost_edges.push_back(std::make_tuple(0,2,5));
    */
  
    auto add_edge = [&](int3 v, int3 delta, int cost/*, int capacity, int capacityRev*/)
      {
        int3 v2 = v+delta;
        if(v2.x < volDim.x &&
           v2.y < volDim.y &&
           v2.z < volDim.z)
          {
            I iv = iii2i(v, volDim);
            I iv2 = iii2i(v2, volDim);
            //const I off = bipartite*data0.size();
            const I off = bipartite*volDim.x*volDim.y*volDim.z;
            cost_edges.emplace_back(iv, iv2+off, 
                                    cost/*, capacity*/);
          
            cost_edges.emplace_back(iv2, iv+off, 
                                    cost/*, capacityRev*/);
          }
      };  
  
  
  
    

    std::list<std::pair<int3,int>> deltaVec;

    if(!bipartite)
      {
        deltaVec.emplace_back(make_int3(1,0,0), 1);
        deltaVec.emplace_back(make_int3(0,1,0), 1);
        deltaVec.emplace_back(make_int3(0,0,1), 1);
    
        const size_t nEdges = volDim.x*volDim.y*volDim.z*deltaVec.size()*2;
        /*      
        // 16 bytes is overhead per list element
        const size_t nBytes = nEdges*(sizeof(std::tuple<I, I, I>)+16);
        const double mb = nBytes/(1024.*1024.);

        #ifdef VERBOSE
        std::cout << "will create approximately " << nEdges << " edges\n";
        std::cout << "will require " << mb << " MB\n";
        #endif
        */
      
        if(nEdges > 10*1000*1000)
          {
            std::cerr << __func__ << " graph is too large (" << nEdges << " edges), exiting ...\n";
	    return cost_edges;
            //exit(-1);
          }
        cost_edges.reserve(nEdges);
      }

    //int64_t sumDiffMass = 0;
    int3 v;
    for(v.z=0; v.z<volDim.z; v.z++)
      for(v.y=0; v.y<volDim.y; v.y++)
        for(v.x=0; v.x<volDim.x; v.x++)
          {
            auto idx = iii2i(v, volDim);
          
            if(bipartite)
              {
                I idx2 = volDim.x*volDim.y*volDim.z;
                int3 v2;
                for(v2.z=0; v2.z<volDim.z; v2.z++)
                  for(v2.y=0; v2.y<volDim.y; v2.y++)
                    for(v2.x=0; v2.x<volDim.x; v2.x++)                  
                      {
                        I3 d = v-v2;
                        cost_edges.emplace_back(idx, idx2++,
                                                std::abs(d.x)+std::abs(d.y)+std::abs(d.z)/*, 
                                                                                           INT_MAX*/);
                      }
              }
            for(auto d : deltaVec)
              {
                //int capacity = data0[idx];
                //int capacityRev = data0[std::min(iii2i(v+d.first), (I)nVolElems-1)];
                //int capacity = INT_MAX;
                //int capacityRev = INT_MAX;
                add_edge(v, d.first, d.second/*, capacity,
                                               capacityRev*/);
              }

            
          }
    return cost_edges;
  }

template<bool bipartite, typename I, typename I3, typename T>
    std::vector<I>
    createFlowGraphVertices(const std::vector<T>& data0,
                            const std::vector<T>& data1,
                            const I3& volDim)
{
  
  std::vector<I> supply_nodes;
  
  const size_t nVolElems = volDim.x*volDim.y*volDim.z;
  const size_t nRealNodes = nVolElems*(1+bipartite);
  supply_nodes.resize(nRealNodes, 0);    

  //int64_t sumDiffMass = 0;
  int3 v;
  for(v.z=0; v.z<volDim.z; v.z++)
    for(v.y=0; v.y<volDim.y; v.y++)
      for(v.x=0; v.x<volDim.x; v.x++)
        {
          auto idx = iii2i(v, volDim);
          
          auto diff = (I)data0[idx]-(I)data1[idx];
          if(bipartite)
            {
              supply_nodes[idx] = data0[idx];
              supply_nodes[nVolElems+idx] = -((I)data1[idx]);
            }
          else            
            supply_nodes[idx] = diff;
          
          //sumDiffMass += diff;
        }
 
  return supply_nodes;
}
    
template<bool bipartite, typename I, typename I3, typename T>
  std::tuple<
  I,
  std::vector<I>,
  std::vector<std::tuple<I, I, I>>,
  std::vector<I>
  >
distFlow(const std::vector<T>& data0,
         const std::vector<T>& data1,
         const I3& volDim,
         fillMassOpts_t<I> fillMassOpts)
{



  
  //std::list<std::tuple<I, I, I>> cost_edges;
  std::vector<std::tuple<I, I, I>> cost_edges =
    createFlowGraphEdges<bipartite, I>(volDim);

  std::vector<I> supply_nodes;    

  //
  // create/destroy mass at the cost of a penalty 
  // (here distance is the same as the max distance through the volume)
  //
  std::vector<I> flowCreator;
  if(fillMassOpts.singleMassGenerator)
    {
      supply_nodes =
        createFlowGraphVertices<bipartite, I>(data0, data1, volDim);
      
      I cost = fillMassOpts.generatorCost(volDim);
      
#ifdef VERBOSE
      std::cout << "GENERATOR COST " << cost << "|" << volDim.x << " " << volDim.y << " " << volDim.z<< std::endl;
#endif
      addGenerator(supply_nodes, cost_edges, 
                   cost, false);
    }
  else
    {     
      //fill2equalizeMass(supply_nodes,fillMassOpts);
      std::vector<I> a, b;
      equalizeSum(a,b, data0, data1);
      supply_nodes =
      createFlowGraphVertices<bipartite, I>(a, b, volDim);
    }
  
  
#ifdef VERBOSE
  std::cout << "there are " << cost_edges.size() << " edges\n";  
#endif
  
  std::vector<I> flow;
  I totalCost;
  std::tie(totalCost, flow)  = mMinCostFlow(supply_nodes, cost_edges);

  assert(flow.size() == cost_edges.size());
#ifndef NDEBUG
  {
    I sum = 0;
    size_t cnt=0;
    for(auto e : cost_edges)
      {
        const auto f = flow[cnt++];
        const auto w = std::get<2>(e);
        sum += f*w;
        //std::cout << "f " << f*w << std::endl;
      }
    if(sum != totalCost)
      std::cout << __func__ << " " << sum << " vs "  << totalCost << std::endl;
    assert(sum == totalCost);
  }
#endif

  //std::cout << "total cost: " << totalCost << std::endl;

#if 0
  supply_nodes.clear();
  cost_edges.clear();

  supply_nodes.resize(8, 0);
  supply_nodes[0] = 1;
  supply_nodes[7] = -1;
  for(size_t i=0; i<supply_nodes.size()-1; i++)
    cost_edges.push_back(std::make_tuple(i, i+1, 1));
  
  std::tie(totalCost, flow)  =  mMinCostFlow(supply_nodes, cost_edges);
  std::cout << "total cost: " << totalCost << std::endl;
#endif

  return std::make_tuple(totalCost, flow, cost_edges, flowCreator);
}

template<typename I, typename T, typename V3>
  I mcf_cost(T data0, T data1, V3 dim, fillMassOpts_t<I> fillMassOpts)
{
  return std::get<0>(distFlow<false, I>(data0, data1, dim, fillMassOpts));
}

#if 0
template<typename I, typename V, typename T, typename V3>
std::tuple<I, std::list<std::tuple<I,I,V>>> mcf_emd(T data0, T data1, V3 dim)
{
  /*
  std::vector<int> flow;
  int totalCost;
  std::list<std::tuple<int, int, int,int>> cost_edges;
  */
  
  auto rslt = distFlow<false, int>
    (data0, data1, dim);
  
  auto totalCost = std::get<0>(rslt);
  auto flow = std::get<1>(rslt);
  auto cost_edges = std::get<2>(rslt);
    

  
  std::map<std::pair<int,int>, int> edges;
  {
    size_t cnt=0;

    for(auto e : cost_edges)
      {
        if(flow[cnt] != 0)
          {
            
            
            std::cout << "mcf_emd from " << std::get<0>(e)
                      << " to " << std::get<1>(e)
                      << " cost " << std::get<2>(e)
              //<< " capacity " << std::get<3>(e)
                      << " flow " << flow[cnt]          
                      << std::endl;
            

            edges[std::make_pair(std::get<0>(e), std::get<1>(e))]
              = flow[cnt];
          }
        /*
          auto from = std::get<0>(e);
          auto to = std::get<1>(e);

          auto fl = t*flow[cnt];

          dataInterp[from] -= fl;
          dataInterp[to] += fl;
        */
        cnt++;
      }
  }
  
  
  std::vector<int> data1_int(data1.begin(), data1.end());
  return std::make_tuple(0, flowShortcuts(data1_int, edges));
    
    //std::list<std::tuple<I,I,V>> rslt;

    //return rslt;
}
#endif

template<typename I, typename V, typename T, typename V3>
std::tuple<I,std::list<std::tuple<I,I,V>>> std_emd(T data0, T data1, V3 dim)
{
  std::vector<int> flow;
  int totalCost;
  std::vector<std::tuple<int, int, int>> cost_edges;
  std::tie(totalCost, flow, cost_edges) = 
    distFlow<true, int>
    (data0, data1, dim, fillMassOpts_t<int>());
  
  const size_t nElems = dim.x*dim.y*dim.z;

  std::list<std::tuple<I,I,V>> rslt;
  {
    size_t cnt=0;
    for(auto e : cost_edges)
      {
        if(flow[cnt] != 0)
          {
            const auto from = std::get<0>(e);
            const auto to = std::get<1>(e)-nElems;
            /*
            std::cout << "std_emd from " << from
                      << " to " << to
                      << " cost " << std::get<2>(e)
                      << " capacity " << std::get<3>(e)
                      << " flow " << flow[cnt]
                      << std::endl;
            */
            if(from != to)
              rslt.emplace_back(from, to, flow[cnt]);

            //edges[std::make_pair(std::get<0>(e), std::get<1>(e))]
            //= flow[cnt];
          }
        cnt++;
      }
  }
  return std::make_tuple(totalCost,rslt);
}

/*
ostream& operator<<(ostream& out, const double2& t)
{
  return out << t.x << " " << t.y;
}
*/
/*
template<typename T>
inline __host__ __device__ double mlength2(const T& a)
{
  return dot(a,a);
}
*/

// Define a type which holds an unsigned integer value 
template<std::size_t> struct int_{};
 
template <class Tuple, size_t Pos>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<Pos> ) {
  out << std::get< std::tuple_size<Tuple>::value-Pos >(t) << ',';
  return print_tuple(out, t, int_<Pos-1>());
}
 
template <class Tuple>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<1> ) {
  return out << std::get<std::tuple_size<Tuple>::value-1>(t);
}
 
template <class... Args>
ostream& operator<<(ostream& out, const std::tuple<Args...>& t) {
  out << '('; 
  print_tuple(out, t, int_<sizeof...(Args)>()); 
  return out << ')';
}

template<typename T>
size_t getNDims()
{
  auto tT = std::type_index(typeid(T));
  if(tT == std::type_index(typeid(double2)))
    return 2;

  assert(false);
  return std::numeric_limits<size_t>::max();
}

template<int idx>
double& mget(double2& d)
{
  assert(idx==0 || idx==1);
  if(idx==0)
    return d.x;
  else
    return d.y;
  
}


template<typename T>
void massign(double2& d, const T& p)
{
  d.x = p.x;
  d.y = p.y;
}

/*
void massign(pcl::PointXY& p, const double2& d)
{
  p.x = d.x;
  p.y = d.y;
}
*/

template<typename T>
struct Point2Less
{
  bool operator() (const T& a, const T& b) const
  {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  }
};


template<typename T0, typename T1, typename T2>
  T0 clamp(const T0& v, const T1& vmin, const T2& vmax)
{
  if(v<vmin)
    return vmin;
  else if(v>vmax)
    return vmax;
  else
    return v;
}


#endif // USE_OR_TOOLS

#endif //__FLOW_HELPER__
