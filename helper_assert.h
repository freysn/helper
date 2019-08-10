#ifndef __HELPER_ASSERT__
#define __HELPER_ASSERT__

#include <sstream>
namespace helper
  {
    template<typename M>
    inline void _assert(const char* expression, const char* file, int line, M message)
    {
      std::stringstream ss;
      ss << "<|" << message << "|>";
      fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d', %s.", expression, file, line, ss.str().c_str());
      abort();
    }
    
    inline void _assert(const char* expression, const char* file, int line)
    {
      _assert(expression, file, line, std::string(""));
    }
 
#ifdef NDEBUG
#define hassert(EXPRESSION) ((void)0)
#define hassertm(EXPRESSION,MESSAGE) ((void)0)
#else
#define hassert(EXPRESSION) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__))
#define hassertm(EXPRESSION,MESSAGE) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__, MESSAGE))
#endif
}
    
#endif // __HELPER_ASSERT__
