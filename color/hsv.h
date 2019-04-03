#ifndef __HSV__
#define __HSV__

template<typename HSV, typename RGB>
#ifdef __CUDACC__
__host__ __device__
#endif
HSV rgb2hsv(RGB in)
{
  typedef decltype(in.x) F;
  
    HSV         out;
    F      min, max, delta;

    min = in.x < in.y ? in.x : in.y;
    min = min  < in.z ? min  : in.z;

    max = in.x > in.y ? in.x : in.y;
    max = max  > in.z ? max  : in.z;

    out.z = max;                                // v
    delta = max - min;
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.y = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
            // s = 0, v is undefined
        out.y = 0.0;
        out.x = NAN;                            // its now undefined
        return out;
    }
    if( in.x >= max )                           // > is bogus, just keeps compilor happy
        out.x = ( in.y - in.z ) / delta;        // between yellow & magenta
    else
    if( in.y >= max )
        out.x = 2.0 + ( in.z - in.x ) / delta;  // between cyan & yellow
    else
        out.x = 4.0 + ( in.x - in.y ) / delta;  // between magenta & cyan

    out.x *= 60.0;                              // degrees

    if( out.x < 0.0 )
        out.x += 360.0;

    return out;
}

template<typename RGB, typename HSV>
#ifdef __CUDACC__
__host__ __device__
#endif
RGB hsv2rgb(HSV in)
{
  typedef decltype(in.x) F;
    F      hh, p, q, t, ff;
    long        i;
    RGB         out;

    if(in.y <= 0.0) {       // < is bogus, just shuts up warnings
        out.x = in.z;
        out.y = in.z;
        out.z = in.z;
        return out;
    }
    hh = in.x;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.z * (1.0 - in.y);
    q = in.z * (1.0 - (in.y * ff));
    t = in.z * (1.0 - (in.y * (1.0 - ff)));

    switch(i) {
    case 0:
        out.x = in.z;
        out.y = t;
        out.z = p;
        break;
    case 1:
        out.x = q;
        out.y = in.z;
        out.z = p;
        break;
    case 2:
        out.x = p;
        out.y = in.z;
        out.z = t;
        break;

    case 3:
        out.x = p;
        out.y = q;
        out.z = in.z;
        break;
    case 4:
        out.x = t;
        out.y = p;
        out.z = in.z;
        break;
    case 5:
    default:
        out.x = in.z;
        out.y = p;
        out.z = q;
        break;
    }
    return out;
}

#endif //__HSV__
