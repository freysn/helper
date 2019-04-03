#include "volData.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <fstream>
#include <vector>
#include <set>
#include <cstring>
#include <climits>
#include <ctime>
#include <cassert>

typedef unsigned int m_uint;

/*
struct ltstr
{
  bool operator()(const m_int3& a, const m_int3& b) const
{
  if(a.x < b.x)
    return true;
  if(a.x > b.x)
    return false;

  if(a.y < b.y)
    return true;
  if(a.y > b.y)
    return false;

  if(a.z < b.z)
    return true;
  
  return false;
}
};
*/

class Item
{
public:
  Item()
  {    
    radius=0;
    center.x = -1;
    center.y = -1;
    center.z = -1;
  }

  bool getCoveredPos(std::vector<m_int3>& pos, m_uint3 dim)
  {
    m_int3 p;
    bool allInsideBounds = true;
    for(p.z=center.z-radius; p.z<=center.z+radius; p.z++)
      for(p.y=center.y-radius; p.y<=center.y+radius; p.y++)
        for(p.x=center.x-radius; p.x<=center.x+radius; p.x++)
          {
            if(p.x < 0 || p.y < 0 || p.z < 0 ||
               p.x >= dim.x || p.y >= dim.y || p.z >= dim.z)
              {
                allInsideBounds = false;
                continue;
              }
            /*
            if((fabs(p.x-center.x) <= lineRadius) + 
               (fabs(p.y-center.y) <= lineRadius) +
               (fabs(p.z-center.z) <= lineRadius) >= 2)
            */
            pos.push_back(p);
          }

    return allInsideBounds;
  }

  //int lineRadius;
  int radius;

  m_int3 center;
};


int main(int argc, char** argv)
{
  const m_uint3 dim(32, 32, 32);
  const m_uint nTimeSteps=128;
  typedef unsigned char volData_t;
  const size_t nElems = dim.x*dim.y*dim.z;
  std::vector<volData_t> volDataBuf(nElems);

  const std::string filename("out/box");
  const m_uint numFixedLen = 4;

  Item item;
  item.radius = 10.;
  item.center.x = 0+item.radius;
  item.center.y = dim.y/2.;
  item.center.z = dim.z/2.;

  int dx = 1;
  for(m_uint t=0; t<nTimeSteps; t++)
    {                  

      std::cout << t << ", center: " << item.center.x << " " << item.center.y << " " << item.center.z << std::endl;

      std::vector<m_int3> pos;
      item.getCoveredPos(pos, dim);
      memset(&volDataBuf[0], 0, sizeof(volData_t)*nElems);
      std::cout << "there are " << pos.size() << " marked elems\n";
      for(auto p : pos)
        {
          //std::cout << p.x << " " << p.y << " " << p.z << std::endl;
          size_t id = p.x+dim.x*(p.y+dim.y*p.z);
          assert(id<volDataBuf.size());
          volDataBuf[id] = 255;
        }
      
      {
        std::stringstream ss;
        ss << setw(numFixedLen) << setfill('0') << t;
        std::string fname = filename+"_UCHAR_"+ss.str()+".raw";
        writeFile(&volDataBuf[0], nElems, fname);      
      }

      item.center.x += dx;

      
      if(item.center.x-item.radius <= 0)
        dx=1;
      else if(item.center.x+item.radius >=dim.x)
        dx=-1;
    }
        
  
  return 0;
}

