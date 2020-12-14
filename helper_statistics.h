#include <numeric>
#include <algorithm>

namespace helper
{
  template<typename RandomAccessIterator>
    typename std::iterator_traits<RandomAccessIterator>::value_type 
    mean(RandomAccessIterator b, RandomAccessIterator e)
    {
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type 
	value_type;
    
      value_type t = std::accumulate(b, e, static_cast<value_type>(0));

      return t / (e-b);    
    }

  template<typename RandomAccessIterator>
    typename std::iterator_traits<RandomAccessIterator>::value_type 
    stdev(RandomAccessIterator b, RandomAccessIterator e)
    {
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type 
	value_type;

      const value_type m = mean(b,e);
      value_type accum = 0;
      std::for_each (b, e, [&](const double d) {
	  accum += (d - m) * (d - m);
	});

      value_type stdev = std::sqrt(accum / ((e-b)-1));
    
      return stdev;
    }
  
}
