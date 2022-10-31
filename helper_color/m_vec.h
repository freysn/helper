#ifndef __M_VEC__
#define __M_VEC__

template <class... Ts> struct m_vec {};

template <class T, class... Ts>
  struct m_vec<T, Ts...> : m_vec<Ts...>
{
#ifdef __NVCC__
  __device__
#endif
 m_vec(T t, Ts... ts) : m_vec<Ts...>(ts...), tail(t)
  {
  }

#ifdef __NVCC__
  __device__
#endif
    m_vec()
  {
  }

  T tail;
};

template <unsigned int k, typename OP, class... Ts>
#ifdef __NVCC__
  __device__
#endif
  void  m_forall(OP& a, m_vec<Ts...>& t)
{  
}

template <unsigned int k=0, typename OP, class T, class... Ts>
#ifdef __NVCC__
  __device__
#endif
  void m_forall(OP& op, m_vec<T, Ts...>& t)
{
  m_vec<Ts...>& base = t;
  op.template operator()<k>(t.tail);
  m_forall<k+1>(op, base);
}

#include "m_vec_typeInterface.h"

//
//
//
/*
  template <unsigned int x, unsigned int k, class... Ts>
#ifdef __NVCC__
  __device__
#endif
  int  m_get(m_vec<Ts...>& t)
{
  return 0;
}

template <unsigned int x, unsigned int k=0, class T, class... Ts>
#ifdef __NVCC__
  __device__
#endif
  T m_get(m_vec<T, Ts...>& t)
{
  m_vec<Ts...>& base = t;
  if(k==x)
    return t.tail;
  //op.template operator()<k>(t.tail);
  else
    return m_get<x, k+1>(base);
}
*/
//
//
//


#endif //__M_VEC__
