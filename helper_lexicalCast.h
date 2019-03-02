#ifndef __LEXICAL_CAST__
#define __LEXICAL_CAST__

#include <sstream>

namespace helper
{
  template<typename to, typename from>
    to lexicalCast(from const &x) {
    std::stringstream os;
    to ret;

    os << x;
    os >> ret;

    return ret;  
  }

  template<typename to, typename from>
    void lexicalCast(to & y, from const &x)
  {
    y = lexicalCast<to>(x);
  }
}

#endif //__LEXICAL_CAST__
