#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


namespace helper
{
  template<typename T>
  auto thrust_vec2raw(T& v)
  {
    return thrust::raw_pointer_cast(&v[0]);
  };

  template<typename T>
  auto thrust_vec2raw_const(const T& v)
  {
    return thrust::raw_pointer_cast(&v[0]);
  };  
};
