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

    template<typename M0, typename M1>
    inline void _assert(const char* expression, const char* file, int line, M0 message0, M1 message1)
    {
      std::stringstream ss;
      ss << "<|" << message0 << "|" << message1<< "|>";
      fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d', %s.", expression, file, line, ss.str().c_str());
      abort();
    }

    template<typename M0, typename M1, typename M2>
    inline void _assert(const char* expression, const char* file, int line, M0 message0, M1 message1, M2 message2)
    {
      std::stringstream ss;
      ss << "<|" << message0 << "|" << message1<< "|" << message2<< "|>";
      fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d', %s.", expression, file, line, ss.str().c_str());
      abort();
    }

    template<typename M0, typename M1, typename M2, typename M3>
    inline void _assert(const char* expression, const char* file, int line, M0 message0, M1 message1, M2 message2, M3 message3)
    {
      std::stringstream ss;
      ss << "<|" << message0 << "|" << message1<< "|" << message2<< "|" << message3<< "|>";
      fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d', %s.", expression, file, line, ss.str().c_str());
      abort();
    }
    
    inline void _assert(const char* expression, const char* file, int line)
    {
      _assert(expression, file, line, std::string(""));
    }
    
#ifdef NDEBUG
// #define hassert(EXPRESSION) ((void)0)
// #define hassertm(EXPRESSION,MESSAGE) ((void)0)
// #define hassertm2(EXPRESSION,MESSAGE0, MESSAGE1) ((void)0)
// #define hassertm3(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2) ((void)0)
// #define hassertm4(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2, MESSAGE3) ((void)0)

#define hassert(EXPRESSION) do { (void)sizeof(EXPRESSION);} while (0)
#define hassertm(EXPRESSION,MESSAGE) do { (void)sizeof(EXPRESSION);} while (0)
#define hassertm2(EXPRESSION,MESSAGE0, MESSAGE1) do { (void)sizeof(EXPRESSION);} while (0)
#define hassertm3(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2) do { (void)sizeof(EXPRESSION);} while (0)
#define hassertm4(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2, MESSAGE3) do { (void)sizeof(EXPRESSION);} while (0)
        
#else

#ifdef EMSCRIPTEN
#define hassert(EXPRESSION) assert(EXPRESSION)
#define hassertm(EXPRESSION,MESSAGE) assert(EXPRESSION)
#define hassertm2(EXPRESSION,MESSAGE0, MESSAGE1) assert(EXPRESSION)
#define hassertm3(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2) assert(EXPRESSION)
#define hassertm4(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2, MESSAGE3) assert(EXPRESSION)
#else
#define hassert(EXPRESSION) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__))
#define hassertm(EXPRESSION,MESSAGE) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__, MESSAGE))
#define hassertm2(EXPRESSION,MESSAGE0, MESSAGE1) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__, MESSAGE0, MESSAGE1))
#define hassertm3(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__, MESSAGE0, MESSAGE1, MESSAGE2))
#define hassertm4(EXPRESSION,MESSAGE0, MESSAGE1, MESSAGE2, MESSAGE3) ((EXPRESSION) ? (void)0 : helper::_assert(#EXPRESSION, __FILE__, __LINE__, MESSAGE0, MESSAGE1, MESSAGE2, MESSAGE3))
#endif // EMSCRIPTEN
#endif // NDEBUG
}
    
#endif // __HELPER_ASSERT__
