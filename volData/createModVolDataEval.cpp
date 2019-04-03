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

//typedef float T_in;
typedef unsigned char T_in;
typedef unsigned char T_out;
//typedef unsigned short T_out;

//const double valueMaxOut = 65535.f;
const double valueMaxOut = 255.f;

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

class Item
{
public:
  Item()
  {    
    type=0;
    radius=0;
    center.x = -1;
    center.y = -1;
    center.z = -1;
  }

  bool getCoveredPos(std::vector<m_int3>& pos, m_uint3 dim)
  {
    m_int3 p;
    for(p.z=center.z-radius; p.z<=center.z+radius; p.z++)
      for(p.y=center.y-radius; p.y<=center.y+radius; p.y++)
        for(p.x=center.x-radius; p.x<=center.x+radius; p.x++)
          {
            if(p.x < 0 || p.y < 0 || p.z < 0 ||
               p.x >= dim.x || p.y >= dim.y || p.z >= dim.z)
              return false;

            if((fabs(p.x-center.x) <= lineRadius) + 
               (fabs(p.y-center.y) <= lineRadius) +
               (fabs(p.z-center.z) <= lineRadius) >= 2)
              pos.push_back(p);
          }

    return true;
  }

  int lineRadius;
  int radius;
  int type;

  m_int3 center;
};

int main(int argc, char** argv)
{
  //m_uint3 dim(720, 320, 320);
  //m_uint3 dim(1024, 1024, 1080);
  m_uint3 dim(1024, 1024, 1024);
  //m_um_int3 fromDim(128, 128, 256);
  //m_um_int3 fromDim(680, 680, 680);
  //m_um_int3 dim(442, 442, 1500);

  srand(time(0));
  
  
  //char* filename = "/data/volumes/volume/utct/oldfield_mouse.raw";
  std::string filename(argv[1]);

  const size_t nElemsIn = dim.x*dim.y*dim.z;
  const size_t nElemsOut = dim.x*dim.y*dim.z;
  
  T_in* fromData = new T_in[nElemsIn];
  //T_in* toDataBuf = new T_in[nElemsOut];
  
  T_out* toData = new T_out[nElemsOut];


  const unsigned int numFixedLen = 3;  

#if 0
  std::stringstream ss;
  ss << setw(numFixedLen) << setfill('0') << i;
      
  std::string baseName = filename+ss.str(); 
  baseName += ".raw";
#else          
  std::string baseName = filename; 
#endif
  std::cout << baseName << std::endl;

  std::ifstream fin;
  fin.open((char*)baseName.c_str(), std::ios::in | std::ios::binary);
  assert(fin.is_open());
  fin.read((char*)fromData, sizeof(T_in)*nElemsIn);
          
  for(int i=0; i<=0; i++)
  //for(int modificationId=0; modificationId<10; modificationId++)
  {               
    std::set<m_int3, ltstr> coveredPos;
    //const unsigned int nElems = 3+rand()%5;
    const unsigned int nElems = 20;

    std::cout << "generate items: " << nElems << std::endl;
    //
    // determine covered spots
    //

    std::cout << "determine covered spots\n";
    {
      int e=0;
      while(e<nElems)
        {
          
          Item item;
                
          item.center.x = rand() % dim.x;
          item.center.y = rand() % dim.y;
          item.center.z = rand() % dim.z;

          //item.radius = 4+rand() % 4;
          item.radius = 8;
          item.lineRadius = 2;

          //item.radius = 2;
          //item.lineRadius = 1;

          std::cout << "e: " << e << " pos: " << item.center.x << " " << item.center.y << " " << item.center.z << std::endl;          

          std::vector<m_int3> pos;
          if(!item.getCoveredPos(pos, dim))
            continue;

          {
            bool oneNonEmpy = false;
          for(std::vector<m_int3>::iterator iter = pos.begin();
              iter != pos.end(); iter++)
            {
              oneNonEmpy = oneNonEmpy || 
                fromData[iter->x + dim.x*(iter->y+dim.y*iter->z)] > 
                32;
                //128;
            }
          if(!oneNonEmpy)
            continue;
          }
                
          {
            bool success = true;
            for(size_t i=0; i<pos.size(); i++)
              {
                if(coveredPos.find(pos[i]) != coveredPos.end())
                  success = false;

                
              }
            if(!success)
              continue;
          }
          assert(!pos.empty());
          coveredPos.insert(pos.begin(), pos.end());                
          e++;
        }
    }

    //
    // determine valid density value
    //
    std::cout << "there are " << coveredPos.size() << " positions\n";
    std::cout << "determine valid density value\n";
    
    T_in density = 255;
    //T_in density = 0;
    
#if 0
    while(true)
      {
        
        density = rand() % 256;

        std::cout << "testing: " << (int) density << std::endl;
        
        bool success = true;

        int smallestDistance = INT_MAX;
        for(std::set<m_int3>::iterator iter = coveredPos.begin();
            iter != coveredPos.end(); iter++)
          {
            assert(iter->x >= 0 && iter->x < dim.x);
            assert(iter->y >= 0 && iter->y < dim.y);
            assert(iter->z >= 0 && iter->z < dim.z);
            int v = fromData[iter->x + dim.x*(iter->y+dim.y*iter->z)];

            int dist = abs(v-density);
            std::cout << dist << std::endl;
            smallestDistance = std::min(smallestDistance, dist);
            if(dist < 10)
              {
                success = false;
                break;
              }
          }

        std::cout << "smallest distance: " << smallestDistance << std::endl;

        if(success)
          break;        
      }
#endif
    std::cout << "density: " << (int) density << std::endl;

    memcpy(toData, fromData, dim.x*dim.y*dim.z*sizeof(T_in));

    std::cout << "copy density: " << (int) density << std::endl;
    for(std::set<m_int3>::iterator iter = coveredPos.begin();
            iter != coveredPos.end(); iter++)
      {
        toData[iter->x + dim.x*(iter->y+dim.y*iter->z)] = density;
      }
           
  
    std::stringstream ss;
    ss << setw(numFixedLen) << setfill('0') << i;
              
    std::string fname = filename+"_UCHAR_"+ss.str()+".raw";
    std::cout << "write file " << fname << std::endl;
    writeFile(toData, dim.x*dim.y*dim.z, 
              (char*)fname.c_str());      
  }

        
  
  return 0;
}

