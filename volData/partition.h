#ifndef __PARTITION__
#define __PARTITION__

#include <vector>
#include <cstring>

template<typename I3>
std::pair<I3, I3> partitionRegular(const size_t& id, const I3& brickGrid, const I3& dataDim)
{
  assert(id < brickGrid.x*brickGrid.y*brickGrid.z);

  I3 id3;
  id3.z = id/(brickGrid.x*brickGrid.y);
  id3.y = (id % (brickGrid.x*brickGrid.y))/brickGrid.x;
  id3.x = id % brickGrid.x;
  assert(id3.x < brickGrid.x);
  assert(id3.y < brickGrid.y);
  assert(id3.z < brickGrid.z);
  
  I3 brickOff;  
  I3 brickDim;

  brickDim.x = dataDim.x/brickGrid.x;
  brickDim.y = dataDim.y/brickGrid.y;
  brickDim.z = dataDim.z/brickGrid.z;

  brickOff.x = brickDim.x*id3.x;
  brickOff.y = brickDim.y*id3.y;
  brickOff.z = brickDim.z*id3.z;


  if(id3.x == brickDim.x-1)
    brickDim.x = dataDim.x-brickOff.x;
  if(id3.y == brickDim.y-1)
    brickDim.y = dataDim.y-brickOff.y;
  if(id3.z == brickDim.z-1)
    brickDim.z = dataDim.z-brickOff.z;  

  return std::make_pair(brickOff, brickDim);
}

template<typename T2, typename T, typename I3>
  void crop(std::vector<T2>& cropOut,
            const I3& cropOff,
            const I3& cropDim,
            const std::vector<T>& data,
            const I3& dim)
{
  cropOut.resize(cropDim.x*cropDim.y*cropDim.z);
  for(size_t z=0; z<cropDim.z; z++)
    for(size_t y=0; y<cropDim.y; y++)
      {
        const size_t dataPos =
          cropOff.x+dim.x*(cropOff.y+y+dim.y*(cropOff.z+z));
        std::copy(data.begin()+dataPos,
                  data.begin()+dataPos+cropDim.x,
                  cropOut.begin()+(cropDim.x*(y+cropDim.y*z)));
      }
}

#endif //__PARTITION__
