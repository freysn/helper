#define M_VEC
#include "volData/vec.h"

typedef double F;
typedef V2<F> float2;
typedef V2<int> int2;
typedef V3<int> int3;

#include "helper/helper_marchingSquares.h"
#include "mergeSegments.h"
#include "recDraw.h"

#include "helper/helper_idx.h"
#include "helper/helper_cimg.h"

#include <random>

int main(const int argc, const char** argv)
{

  std::vector<std::vector<float2> > lines;
  int width=512;
  int height=width;

  std::vector<F> data;
  if(argc == 2)
    {
      std::vector<uint8_t> data_ui8;
      V3<int> dim;
      int nChannels;
      helper::cimgRead(data_ui8, dim, nChannels, argv[1]);

      assert(nChannels==1);
      width = dim.x;
      height = dim.y;
      data.resize(data_ui8.size());
      for(size_t i=0; i<data.size(); i++)
	data[i] = data_ui8[i]/255.;
    }
  else
    {
      std::mt19937 gen(1337);
      std::uniform_real_distribution<> dis(0, 1);
      
      data.resize(width*height);

      for(int y=0; y<height; y++)
	for(int x=0; x<width; x++)
	  {
	  data[x+y*width] =
	    
	    (length(V2<F>(x,y)-V2<F>(width/2, height/2))/*-0.2*width*/)
	    //(x-y+0.1)
	    /
	    length(V2<F>(width, height));

	  data[x+y*width] = dis(gen);
	  }
    }

  const auto borderValue =
    std::max(2* (*std::max_element(data.begin(), data.end())), 0.);

  
  //std::nexttoward(*std::max_element(data.begin(), data.end()),
  //		    std::numeric_limits<F>::max());
    

  std::cout << "border value: " << borderValue << std::endl;
  
  V2<int> p;
  for(p.y=3; p.y<height-1; p.y++)
    for(p.x=4; p.x<width-1; p.x++)
      {
	//if(!lines.empty())
	//continue;
	
	auto addPoint = [&lines](auto p)
	  {
	    if(false)
	      {
	    if(lines.empty())
	      lines.resize(lines.size()+1);
	    lines.back().emplace_back(p);
	      }
	  };

	auto texLookup =
	  [&data, & width, &height, &borderValue]
	  (auto p)
	  {
	    const V2<int> dim(width, height);
	    if(helper::iiWithinBounds(p,dim))
	      return data[helper::ii2i/*_clamp*/(p, dim)];
	    else
	      {
		typedef std::remove_reference<decltype(data[0])>::type E;
		//return std::numeric_limits<E>::max();
		return borderValue;
	      }
	  };

	V4<F> values;

	F minV = std::numeric_limits<F>::max();
	F maxV = std::numeric_limits<F>::min();
	
	for(size_t i=0; i<4; i++)
	  {
	    values[i] = texLookup(p+helper::getVCodeOffset<decltype(p)>(i));
	    minV = std::min(minV, values[i]);
	    maxV = std::max(maxV, values[i]);
	  }

	
	if(maxV-minV <= helper::eps<F>())
	  continue;

	
	const auto isovalue =
	  0.5*(minV+maxV);
	//0.2;

	std::cout << p << "--------------value range: " << minV << " " << maxV << " , isovalue: " << isovalue << std::endl;
	
	helper::walkIsoline(addPoint,
			    p,
			    isovalue,
			    texLookup);
      }
  
  /*
  recLinesMS(lines,
             width, 
             height,
             data);
  */

    std::cout << "there are " << lines.size() << " lines\n";
  for(size_t i=0; i<lines.size(); i++)
    {
      for(size_t j=0; j<lines[i].size(); j++)
        {
          std::cout << "(" << lines[i][j].x << ", "
                    << lines[i][j].y << ") ";
          
        }
      std::cout << std::endl;
      }

  /*
  std::vector<std::list<float2> > newLines;
  mergeSegments(newLines, lines);
  std::cout << "there are " << newLines.size() << " new lines\n";
  */
  
  //drawLines(std::string("lines.pdf"), newLines);
  drawLines(std::string("lines.pdf"), lines);
  
  return 0;
}
