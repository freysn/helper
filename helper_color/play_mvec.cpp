#include <iostream>

struct float4
{
  float x;
  float y;
  float z;
  float w;
};

#include "m_vec.h"


int main(int, char**)
{
  float4 f4;
  f4.x = 234;
  f4.y = 23;
  f4.z = 12.2;
  f4.w = 23.2;
  m_vec<int, double, float4, double> test(5,6.6,f4,12.5);
  std::cout << m_get<3>(test) << std::endl;
  
  return 0;
}
