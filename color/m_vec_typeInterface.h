
template<size_t k>
#ifdef __NVCC__
__device__
#endif
float m_get(const float4& f4)
{
  if(k==0)
    return f4.x;
  else if(k==1)
    return f4.y;
  else if (k==2)
    return f4.z;
  else
    return f4.w;
}

template<size_t k>
#ifdef __NVCC__
__device__
#endif
void m_assign(float4& f4, float v)
{
  if(k==0)
    f4.x = v;
  else if(k==1)
    f4.y = v;
  else if (k==2)
    f4.z = v;
  else
    f4.w = v;
}
