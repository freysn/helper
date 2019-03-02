#ifndef __HELPER_SAT__
#define __HELPER_SAT__

// summed area table aka integral image

namespace helper
{
  template<typename T, typename DIM>
  void genSAT(T& data, const DIM dim)
  {
    for(size_t y=0; y<dim.y; y++)
      for(size_t x=0; x<dim.x; x++)
	{
	  const bool bx = (x>0);
	  const bool by = (y>0);
	  const auto a = bx ? data[x-1+y*dim.x]:0;
	  const auto b = by ? data[x+(y-1)*dim.x]:0;
	  const auto c = (bx && by) ? data[x-1+(y-1)*dim.x]:0;
	
	  //b = (j-1>=0)?sat[i][j-1]:0;
	  //c = ((i-1>=0)&&(j-1>=0))?sat[i-1][j-1]:0;
	  //m = ((i-1>=0)&&(j-1>=0))?matrix[i-1][j-1]:0;
	  // ACTUAL FORMULA FOR SUMMED AREA TABLE
	  //sat[i][j] = m + a + b - c;
	  data[x+y*dim.x] += a+b-c;
	}
  }

  template<typename T, typename DIM>
  auto patchSumFromSAT(DIM from, DIM to, T& sat, const DIM dim)
  {
    assert(to.x > 0 && to.y > 0);
    to.x--;
    to.y--;

    const bool bx = (from.x>0);
    const bool by = (from.y>0);

    return
      sat[to.x+dim.x*to.y]
      + ((bx&&by) ? sat[from.x-1+dim.x*(from.y-1)] : 0)
      - (bx ? sat[from.x-1+dim.x*to.y] : 0)
      - (by ? sat[to.x+dim.x*(from.y-1)] : 0);
  }


  template<typename DIM>
  void testSAT(const DIM dim)
  {
    std::vector<size_t> data(dim.x*dim.y);

    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis(0, 83);


    for(auto & e : data)
      e = dis(gen);

    auto sat = data;
    genSAT(sat, dim);


    for(size_t i=0; i<100; i++)
      {
	std::uniform_real_distribution<> disPatch(1, 32);

	DIM patchDim;
	patchDim.x = disPatch(gen);
	patchDim.y = disPatch(gen);


	std::uniform_real_distribution<> disFromX(0, dim.x-patchDim.x);
	std::uniform_real_distribution<> disFromY(0, dim.y-patchDim.y);

	DIM from(0,0,0,0);
	from.x = disFromX(gen);
	from.y = disFromY(gen);

	DIM to = from+patchDim;

	assert(to.x <= dim.x);
	assert(to.y <= dim.y);

	const auto a = patchSumFromSAT(from, to, sat, dim);

	size_t b=0;
	for(const auto yi : helper::range_n(patchDim.y))
	  for(const auto xi : helper::range_n(patchDim.x))
	    {
	      b += 
		data[from.x+xi+dim.x*(from.y+yi)];
	    }


	std::cout << a << " vs " << b << " patchDim: " << patchDim << " " << from << " " << to << std::endl;
	assert(a==b);
      
      }
  
  }

}

#endif //__HELPER_SAT__
