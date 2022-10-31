#ifndef __CM_GRAY__
#define __CM_GRAY__

template<typename F>
std::vector<V3<F>> cm_gray()
{
  std::vector<V3<F>> v(256);
  for(size_t i=0; i<v.size(); i++)
    {
      const F e = static_cast<F>(i)/(v.size()-1);
      v[i] = V3<F>(e,e,e);
    }
  return v;
}

#endif //__CM_GRAY__
