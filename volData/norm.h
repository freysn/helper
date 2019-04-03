#ifndef __NORM__
#define __NORM__

template <typename T>
static double normt(const T t)
{
#if __cplusplus >= 201103L
  using namespace std;
#else
  using namespace boost;
#endif
  if(/*std::*/is_integral<T>())
    {
      assert(/*std::*/is_unsigned<T>());
      const double mv = std::numeric_limits<T>::max();
      const double tv = t;
      return tv/mv;
    }
  else
    return t;
}

template <typename TO, typename T>
  static std::vector<TO> normt(const std::vector<T>& t)
{
  std::vector<TO> out(t.size());
  for(size_t i=0; i<t.size(); i++)
    out[i] = normt(t[i]);
  return out;
}

template <typename T>
static T denormt(const double t)
{
  #if __cplusplus >= 201103L
  using namespace std;
#else
  using namespace boost;
#endif
  if(/*std::*/is_integral<T>())
    {
      //clamp to allowed value range
      T v = std::max(0.,
                     std::min(
                              t*std::numeric_limits<T>::max()+.5,
                              (double) std::numeric_limits<T>::max()));
      assert(v>=0 && v<= std::numeric_limits<T>::max());
      return v;
    }
  else
    return t;
}

template <typename T>
static void denormt(T& out, const double t)
{
  out = denormt<T>(t);
}

template <typename T>
static T denormt(const float4& f4)
{
  T out;
  denormt(out.x, f4.x);
  denormt(out.y, f4.y);
  denormt(out.z, f4.z);
  denormt(out.w, f4.w);
  return out;     
}

template <typename T>
static void denormt(T& out, const float4& f4)
{
  out = denormt<T>(f4);
}
 
template <typename T>
static T denormt(const std::vector<float4>& f4)
{
     
  T out;
  out.resize(f4.size());
  for(size_t i=0; i<out.size(); i++)
    denormt(out[i], f4[i]);
  return out;     
}

#endif //__NORM__
