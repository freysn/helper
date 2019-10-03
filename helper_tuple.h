#ifndef __HELPER_TUPLE__
#define __HELPER_TUPLE__
  
template<class TupType, size_t... I>
std::ostream& tuple_print(std::ostream& os,
			  const TupType& _tup, std::index_sequence<I...>)
{
  os << "(";
  (..., (os << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
  os << ")";
  return os;
}

template<class... T>
std::ostream& operator<< (std::ostream& os, const std::tuple<T...>& _tup)
{
  return tuple_print(os, _tup, std::make_index_sequence<sizeof...(T)>());
}

#endif //__HELPER_TUPLE__
