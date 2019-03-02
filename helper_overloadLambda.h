#ifndef __HELPER_OVERLOAD_LAMBDA__
#define __HELPER_OVERLOAD_LAMBDA__

namespace helper
{
  template <class... Fs>
struct overload;

template <class F0, class... Frest>
struct overload<F0, Frest...> : F0, overload<Frest...>
{
    overload(F0 f0, Frest... rest) : F0(f0), overload<Frest...>(rest...) {}
    
    using F0::operator();
    using overload<Frest...>::operator();
};

template <class F0>
struct overload<F0> : F0
{
    overload(F0 f0) : F0(f0) {}
    
    using F0::operator();
};

template <class... Fs>
auto make_overload(Fs... fs)
{
    return overload<Fs...>(fs...);
}

  /*
  int main(int argc, char* argv[])
  {
    int y = 123;

    auto f = make_overload(
        [y] (int x) { std::cout << "int x==" << x << std::endl; },
        [y] (char *cp) { std::cout << "char *cp==" << cp << std::endl; });
    f(argc);
    f(argv[0]);
  }
  */
};

#endif //__HELPER_OVERLOAD_LAMBDA__
