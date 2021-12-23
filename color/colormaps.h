#ifndef __COLORMAPS__
#define __COLORMAPS__

#include <vector>

std::vector<V3<float>>
#ifndef __CUDACC__
get_colormap_brewer0
#else
get_colormap_brewer0_ccc
#endif
()
{
  const V3<float> col0 = make_V3<float>(228,26,28)/255.f;
  const V3<float> col1 = make_V3<float>(55,126,184)/255.f;
  const V3<float> col2 = make_V3<float>(77,175,74)/255.f;
  const V3<float> col3 = make_V3<float>(152,78,163)/255.f;
  const V3<float> col4 = make_V3<float>(255,127,0)/255.f;
  //const V3<float> col5 = make_V3<float>(255,255,51)/255.f;
  //const V3<float> col5 = make_V3<float>(223,255,0)/255.f;
  const V3<float> col5 = make_V3<float>(0, 190, 190)/255.f;
  const V3<float> col6 = make_V3<float>(166,86,40)/255.f;
  const V3<float> col7 = make_V3<float>(247,19,191)/255.f;
  const V3<float> col8 = make_V3<float>(247,91,19)/255.f;
  const V3<float> col9 = make_V3<float>(91,19,47)/255.f;

  std::vector<V3<float>> cols(10);
  cols[0] = col0;
  cols[1] = col1;
  cols[2] = col2;
  cols[3] = col3;
  cols[4] = col4;
  cols[5] = col5;
  cols[6] = col6;
  cols[7] = col7;
  cols[8] = col8;
  cols[9] = col9;
  return cols;
}

#ifdef __CUDACC__
const int nColorsMapBrewer0 = 10;
__constant__ V3<float> c_colormap_brewer0[nColorsMapBrewer0];

void init_colormap_brewer0()
{
  std::vector<V3<float>> c = get_colormap_brewer0_ccc();
  assert(nColorsMapBrewer0 == c.size());
  checkCudaErrors(cudaMemcpyToSymbol(c_colormap_brewer0, &c[0],
                                     sizeof(V3<float>)*nColorsMapBrewer0));
}
#endif

typedef struct {
    double r;       // percent
    double g;       // percent
    double b;       // percent
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // percent
    double v;       // percent
} hsv;


template<typename HSV, typename RGB>
#ifdef __CUDACC__
__device__
#endif
HSV rgb2hsv(RGB in)
{
    hsv         out;
    float      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
            // s = 0, v is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}

template<typename RGB, typename HSV>
#ifdef __CUDACC__
__device__
#endif
rgb hsv2rgb(HSV in)
{
    float      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}



template<typename F3, typename F>
#ifdef __CUDACC__
__device__
#endif
F3 ottlcolormap_hsv(const F& t)
{
  hsv in;
  in.h = 280.f-t*300.f;
  if(in.h < 0.f)
    in.h += 360.f;
  in.s = 0.89;
  in.v = 0.89;
  rgb out = hsv2rgb<rgb>(in);

    F3 col;
    col.x = out.r;
    col.y = out.g;
    col.z = out.b;
  
  /*
  F3 col;
  
  col.x = t;
  col.y = 1.-2.*fabs(t-0.5);
  col.z = 1.-t;

  if(t<0.5f)
    col.z += 0.5f*(0.5f-t);
  */

  /*
  col.x = max(2.f*t-1.f, 0.15f);  
  col.y = 1.f-2.f*abs(0.5-t);
  col.z = max(1.f-2.f*t, 0.15f);
  */
  return col;
}


template<typename F3, typename F/*, typename I*/>
#ifdef __CUDACC__
__device__
#endif
  F3 colormap_brewer0(const F& t/*, const I& nSlots*/)
{
#ifndef __CUDACC__    
  std::vector<V3<float>> c_colormap_brewer0 = get_colormap_brewer0();
  const int nColorsMapBrewer0 = c_colormap_brewer0.size();
  //std::cout << t << " vs " << c_colormap_brewer0.size() << std::endl;
  assert(t >= 0 && t <= c_colormap_brewer0.size()-1.);
  using namespace std;  
#endif
  int idx0 = t;
  int idx1 = min(idx0+1, nColorsMapBrewer0-1);
  V3<float> c0 = c_colormap_brewer0[idx0];
  V3<float> c1 = c_colormap_brewer0[idx1];

  return (t-idx0)*c1+(idx1-t)*c0;
}


template<typename F3, typename F>
#ifdef __CUDACC__
__device__
#endif
  F3 colormap1(const F& t)
{
  return colormap_brewer0<F3>(t);
}

struct Col0
{
public:
  const int n = 6;

  template<typename F3, typename F>
#ifdef __CUDACC__
    __device__
#endif
    F3 getNorm(const F& tNorm)
  {
    const F t = tNorm*(n-1);
    int idx0 = t;
    int idx1 = idx0+1;
    F3 c0 = colorswitch0<F3>(idx0);
    F3 c1 = colorswitch0<F3>(idx1);
    return (t-idx0)*c1+(idx1-t)*c0;
  }
  
template<typename F3, typename F>
#ifdef __CUDACC__
__device__
#endif
  F3 colorswitch0(const F& t)
{  
  F3 out;
  out.x = out.y = out.z = 0.1;
  switch(t)
    {
    case 0:
      out.x = 0.95;
      out.y = 0.22;
      out.z = 0.14;
      break;
    case 1:
      out.x = 1.;
      out.y = 0.72;
      out.z = 0.14;
      break;
    case 2:
      out.x = 0.35;
      out.y = 0.69;
      out.z = 0.16;
      break;
    case 3:
      out.x = 0.19;
      out.y = 0.89;
      out.z = 0.78;
      break;
    case 4:
      out.x = 0.20;
      out.y = 0.36;
      out.z = 0.95;
      break;
    case 5:
      out.x = 0.68;
      out.y = 0.4;
      out.z = 0.65;
      break;

    default:
      break;
    };
  return out;
}
};

template<typename F3, typename F>
#ifdef __CUDACC__
__device__
#endif
  F3 colormapNorm0(const F& t)
{
  Col0 col;
  return col.getNorm<F3>(t);
}

#endif //__COLORMAPS__
