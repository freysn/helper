#ifndef __SHEPARD_LOCALLY_AFFINE__
#define __SHEPARD_LOCALLY_AFFINE__

#include <vector>
#include "splitStr.h"
#include <cassert>
#include "helper_idx.h"

#include <dlib/optimization.h>

#ifdef USE_OMP
#include <omp.h>
#endif

template<typename T>
T shepardLocallyAffine_g(T p, T l0, T l1, T a, T b, T c)
{
  return a*(p.x-l1.x)+b*(p.y-l1.y)+c*(p.z-l1.z)+l0;
}

// This function is the "residual" for a least squares problem.   It takes an input/output
// pair and compares it to the output of our model and returns the amount of error.  The idea
// is to find the set of parameters which makes the residual small on all the data pairs.

typedef dlib::matrix<double,12,1> input_vector;
typedef dlib::matrix<double,9,1> parameter_vector;

double shepardLocallyAffine_residual (
                                      const input_vector& data,
                                      const parameter_vector& params
)
{
  const double3 l0i = make_double3(data(0), data(1), data(2));
  const double3 l0j = make_double3(data(3), data(4), data(5));
  const double3 l1i = make_double3(data(6), data(7), data(8));
  const double3 l1j = make_double3(data(9), data(10), data(11));
    
  const double3 g = shepardLocallyAffine_g(l1j,
                                           l0i,
                                           l1i,
                                           make_double3(params(0), params(1), params(2)),
                                           make_double3(params(3), params(4), params(5)),
                                           make_double3(params(6), params(7), params(8))
                                           );
  const double g2 = length2(g-l0j);

  return std::sqrt(g2/length2(l1i-l1j));
}

template<typename T, typename I3, typename P>
std::vector<T>
  shepardLocallyAffine(const std::vector<T>& in, const I3& dim,
                       const std::vector<P>& points0,
                       const std::vector<P>& points1)
{

  assert(!points0.empty());
  std::vector<T> out(in.size());

  assert(points0.size()==points1.size());

  const double maxDim = (double)std::max(dim.x-1, std::max(dim.y-1, dim.z-1));

  auto normPos = [&maxDim](double3 p)
    {
      return p/maxDim;
    };
  /*
  auto denormPos = [&maxDim](double3 p)
    {
      return p*maxDim;
    };
  */
  

  //
  // DETERMINE COEFFICIENTS
  //
  std::cout << "determine coefficients\n";
  std::vector<std::tuple<double3, double3, double3>> coeffs(points0.size());
  {    
    using namespace dlib;
    auto strat = objective_delta_stop_strategy(1e-6);

    const bool verbose = false;
    if(verbose)
      strat.be_verbose();

    
  for(size_t i=0; i<coeffs.size(); i++)
    {
      parameter_vector x;
      x = 1;
      std::vector<input_vector> data_samples;

      for(size_t j=0; j<points0.size(); j++)
        {
          if(j==i)
            continue;
          const auto l0i = points0[i];
          const auto l0j = points0[j];
          const auto l1i = points1[i];
          const auto l1j = points1[j];

          input_vector v;
          v(0)= l0i.x;
          v(1)= l0i.y;
          v(2)= l0i.z;

          v(3)= l0j.x;
          v(4)= l0j.y;
          v(5)= l0j.z;

          v(6)= l1i.x;
          v(7)= l1i.y;
          v(8)= l1i.z;

          v(9)= l1j.x;
          v(10)=l1j.y;
          v(11)=l1j.z;
          
          /*
          l0i.y, l0i.z,
            l0j.x, l0j.y, l0j.z,
            l1i.x, l1i.y, l1i.z,
            l1j.x, l1j.y, l1j.z
          */
          //#warning "complete conversion"
          data_samples.push_back(v);
        }
      
      solve_least_squares(strat, 
                          shepardLocallyAffine_residual,
                          //residual_derivative,
                          derivative(shepardLocallyAffine_residual),
                          data_samples,
                          x);

      // Now x contains the solution.  If everything worked it will be equal to params.
      if(verbose)
        std::cout << "inferred parameters: "<< trans(x) << std::endl;

      std::get<0>(coeffs[i]).x = x(0);
      std::get<0>(coeffs[i]).y = x(1);
      std::get<0>(coeffs[i]).z = x(2);

      std::get<1>(coeffs[i]).x = x(3);
      std::get<1>(coeffs[i]).y = x(4);
      std::get<1>(coeffs[i]).z = x(5);

      std::get<2>(coeffs[i]).x = x(6);
      std::get<2>(coeffs[i]).y = x(7);
      std::get<2>(coeffs[i]).z = x(8);      
    }
  }


  //
  // COMPUTE MORPH
  //
  std::cout << "compute morph\n";

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for(size_t i=0; i<in.size(); i++)
    {
      auto v = i2iii(i, dim);
      const double3 vf = make_double3(v.x, v.y, v.z);
      
      double sumd = 0.;
      double3 sumv = make_double3(0., 0., 0.);
      
      for(size_t j=0; j<points0.size();j++)
        {
          const auto p = points0[j];
          const double3 pf = make_double3(p.x, p.y, p.z);
          
          double d2 = pow(length(normPos(pf-vf)), 4.);
          d2 = std::max(d2, 1e-20);
          double invd2 = 1./d2;
          
          sumv += invd2*(shepardLocallyAffine_g(vf, pf,
                                                make_double3(points1[j].x, points1[j].y, points1[j].z),
                                                std::get<0>(coeffs[j]),
                                                std::get<1>(coeffs[j]),
                                                std::get<2>(coeffs[j])));
          sumd += invd2;
        }
      const double3 pf = sumv/sumd;

      double value = 0.;

      if(false)
        {
      I3 pi;
      pi.x = pf.x+0.5;
      pi.y = pf.y+0.5;
      pi.z = pf.z+0.5;

      value = in[iii2i_clamp(pi, dim)];
        }
      else
        value = trilinear(vec3<float>(pf.x,
                                      pf.y,
                                      pf.z),
                          &in[0], dim);
      
      //std::cout << "lookup pos for " << v << " - " << pi << "| " << (int)value << std::endl;
      
      out[i] = value;
    }

  return out;
}


#endif //__SHEPARD_LOCALLY_AFFINE__
