#ifndef __HELPER_SETS__
#define __HELPER_SETS__

namespace helper
{
  struct Counter
  {
    struct value_type { template<typename T> value_type(const T&) { } };
    void push_back(const value_type&) { ++count; }
    size_t count = 0;
  };
  
  template<typename T>
  auto setIntersection(const std::vector<T>& a, const std::vector<T>& b)
  {
    std::vector<T> out;
    //if(!a.empty() && !b.empty())
      std::set_intersection(a.begin(), a.end(),
			    b.begin(), b.end(),
			    std::back_inserter(out));
    return out;
  }
  
  template<typename T>
  auto setUnion(const std::vector<T>& a, const std::vector<T>& b)
  {
    std::vector<T> out;
    std::set_union(a.begin(), a.end(),
		   b.begin(), b.end(),
		   std::back_inserter(out));
    return out;
  }
  
  template<typename T>
  auto setUnion(const std::vector<T>& a, const T& b)
  {
    return setUnion(a, std::vector<T>(1,b));
  }
  
  template<typename T>
  auto setDifference(const std::vector<T>& a, const std::vector<T>& b)
  {
    std::vector<T> out;
    std::set_difference(a.begin(), a.end(),
			b.begin(), b.end(),
			std::back_inserter(out));
    return out;
  }
  
  template<typename T>
  auto setDifference(const std::vector<T>& a, const T& b)
  {
    return setDifference(a, std::vector<T>(1,b));
  }

  
  // template<typename T>
  // auto setInsert(std::vector<T> a, const T& b)
  // {
  //   auto it = std::lower_bound(a.begin(), a.end(), b);
    
  //   //Returns an iterator pointing to the first element in the range [first, last) that is not less than (i.e. greater or equal to) value, or last if no such element is found. 
  //   if(it == members.end() || *it != b)
  //     {
  // 	auto it2 = a.insert(it, b);
  // 	assert(it2==a.begin() || *(it2-1) < *it2);
  // 	assert(it2+1==a.end() || *(it2+1) > *it2);
  //     }
    
  //   return a;
  // }
}

#endif //__HELPER_SETS__
