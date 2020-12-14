#ifndef __HELPER_COLOR__
#define __HELPER_COLOR__

namespace helper
{

  template<typename COL=V4<double>>
  auto hex2rgba(const std::string s)
  {
    V4<int> c;
    const auto n=sscanf(s.c_str(), "%02x%02x%02x%02x", &c.x, &c.y, &c.z, &c.w);
    hassertm(n==4, n);
    if(n!=4)
      return COL(-1,-1,-1,-1);
    else
      return COL(c)/255.;
  }
}

#endif //__HELPER_COLOR__
