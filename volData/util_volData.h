#ifndef __UTIL_VOL_DATA__
#define __UTIL_VOL_DATA__

template<typename T_out, typename T_in>
  T_out lcast(const T_in& in)
{
  std::stringstream ss;
  ss << in;
  T_out out;
  ss >> out;
  return out;
}

template<typename T_out, typename T_in>
  void lcast(T_out& out, const T_in& in)
{
  out = lcast<T_out>(in);
}

template<typename V, typename OP>
V binOp_v3(const V& a, const V& b, OP op)
{
  V out;
  out.x = op(a.x, b.x);
  out.y = op(a.y, b.y);
  out.z = op(a.z, b.z);
  return out;
}


std::ostream& operator<<(std::ostream& out, const float3& t)
{
  return out << t.x << " " << t.y << " " << t.y;
}

#endif //__UTIL_VOL_DATA__
