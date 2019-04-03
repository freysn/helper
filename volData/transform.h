#ifndef __VOL_DATA_TRANSFORM__
#define __VOL_DATA_TRANSFORM__

#include "helper_idx.h"

#include <vector>
#include <cassert>
#include "volData.h"
#include "points2vol.h"

const double p0=0.01;
const double p1=8.;

const double q0=0.1;
const double q1=5.6;
/*
double3 transformFunc0(double3 p)
{
  const double3 v =
    make_double3((std::abs(p.x-0.5)*2),
                 (std::abs(p.y-0.5)*2),
                 (std::abs(p.z-0.5)*2));
                    
                    
  p.x = std::pow(p.x, 0.6+v.y*1.4);
  //p.y = std::pow(p.y, 0.75+v.z*1.11);
  //p.z = std::pow(p.z, 0.7+v.x*1.105);
                    
  return p;
}
*/

double3 transformFunc1(double3 p)
{

  //assert(false);


  
  
  //p.x = std::pow(p.x, p0+p.y*p1);  
  //p.z = std::pow(p.z, q0+(1.-p.x)*q1);
  
  //p.x-=0.5;    
  //p.x = (p.x-0.75)*4.;
  //p.x*=0.5;
  
  
  
  //p.y = std::pow(p.y, 0.75+v.z*1.11);
  //p.z = std::pow(p.z, 0.7+v.x*1.105);

  //p = 2.*p-make_double3(1., 1., 1.);

  p=p/0.6;
  p.x= pow(p.x, 1./(10.*(p.y-0.5)*(p.y-0.5)+0.4));
  p.z= pow(p.z, 1./(2.*(p.y-0.6)*(p.y-0.6)+0.6));
  
  return p;
}

double3 transformFunc1_inv(double3 p)
{
  
  //looks cool
  /////////p.x = (log(p.y)/log(p.x) - p0)/p1;

  //p.z = std::pow(p.z, 1./(q0+(1.-p.x)*q1));
  //p.x=std::pow(p.x, 1./(p0+p.y*p1));
  
  //p.x+=0.5;
  //p.x = p.x-0.5
  //p.x = p.x*0.25+0.75;

  //double3 p2 = 2.*p-make_double3(1., 1., 1.);
  p.x= pow(p.x, 14.*(p.y-0.5)*(p.y-0.5)+0.4);
  p.z= pow(p.z, 2.*(p.y-0.6)*(p.y-0.6)+0.6);

  p=p*0.6;
  
  //p = (p+make_double3(1., 1., 1.))/2.;
  
  //p.y = std::pow(p.y, 0.75+v.z*1.11);
  //p.z = std::pow(p.z, 0.7+v.x*1.105);




  
                    
  return p;
}

template<typename F3, typename I3>
double3 normDim(const F3& v, const I3& dim)
{
  double3 p;
  p.x=p.y=p.z=0.;

  if(dim.x > 1)
    p.x = static_cast<double>(v.x)/(dim.x-1);
  if(dim.y>1)
    p.y = static_cast<double>(v.y)/(dim.y-1);
  if(dim.z>1)
    p.z = static_cast<double>(v.z)/(dim.z-1);
  return p;
}

template<typename F3, typename F3_2, typename I3>
F3 denormDim(F3_2 p, I3 dim)
{
  F3 v;
  v.x = p.x*(dim.x-1);
  v.y = p.y*(dim.y-1);
  v.z = p.z*(dim.z-1);

  return v;
}



template<typename T, typename I3, typename F>
  std::vector<T> transform_back(const std::vector<T>& in, const I3& dim, F func)
{
  assert(dim.x*dim.y*dim.z == in.size());
  std::vector<T> out(in.size());

  I3 v;
  for(v.z=0; v.z<dim.z; v.z++)
    for(v.y=0; v.y<dim.y; v.y++)
      for(v.x=0; v.x<dim.x; v.x++)
        {
          double3 p = normDim(v, dim);

          const double3 p2 = func(p);

          double value = 0.;

          if(false)
            {
          I3 v2;
          v2.x = std::max(0, std::min((int)(p2.x*dim.x+0.5), dim.x-1));
          v2.y = std::max(0, std::min((int)(p2.y*dim.y+0.5), dim.y-1));
          v2.z = std::max(0, std::min((int)(p2.z*dim.z+0.5), dim.z-1));
          value = in[iii2i(v2, dim)];
            }
          else
            {
              auto a = denormDim<vec3<float>>(p2, dim);
              /*
                value =
                trilinear(vec3<float>(p2.x*(dim.x-1),
                                      p2.y*(dim.y-1),
                                      p2.z*(dim.z-1)),
                                &in[0], dim);
              */
              value = trilinear(a, &in[0], dim);
            }
          
          out[iii2i(v, dim)] = value;
        }
  return out;
}










template<typename T, typename I3, typename F>
  std::vector<T> transform(const std::vector<T>& in, const I3& dim, F func)
{
  assert(dim.x*dim.y*dim.z == in.size());
  std::vector<double> out(in.size(), 0.);

  std::vector<std::pair<double3, double>> weightedPoints;
  
  I3 v;  
  for(v.z=0; v.z<dim.z; v.z++)
    for(v.y=0; v.y<dim.y; v.y++)
      for(v.x=0; v.x<dim.x; v.x++)
        {
          double3 p = normDim(v, dim);
          double3 p2 = func(p);

          p2 = denormDim<double3>(p2, dim);

          p2.x = std::max(0. ,std::min(p2.x, (double)dim.x-1.));
          p2.y = std::max(0. ,std::min(p2.y, (double)dim.y-1.));
          p2.z = std::max(0. ,std::min(p2.z, (double)dim.z-1.));
          
          weightedPoints.emplace_back(p2, in[iii2i(v, dim)]);          
        }

  splatWeightedPointsTri(out, weightedPoints, dim);


  std::vector<T> out8(out.size());

  for(size_t i=0; i<out8.size(); i++)
    out8[i] = std::max(0, std::min((int)(out[i]+0.5), 255));
    
  return out8;
}
#endif //__VOL_DATA_TRANSFORM__
