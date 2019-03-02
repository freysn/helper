#include "helper_ThreadHandler.h"
#include <iostream>

bool func()
{
  std::cout << "hello\n";
  return true;
}

int main(int argc, const char** argv)
{
  helper::helper_ThreadHandler<bool> th;
  th.run(func);

  std::cout << "returned with " << th.get() << std::endl;
  
  return 0;
}
