#define M_VEC
#include "volData/vec.h"

#include "helper/helper_util.h"
#include <algorithm>
#include <iostream>

int main(int argc, char** argv)
{
  std::vector<size_t> values(1024);

  std::iota(values.begin(), values.end(), static_cast<size_t>(0));

  const size_t nSelect = 6;
  const auto selected = helper::selectDifferentEntries<double>(values, [](size_t a, size_t b) {return std::max(a,b)-std::min(a,b);}, nSelect);

  std::cout << selected.size() << " selected, " << nSelect << " where requested\n";

  for(const auto e : selected)
    std::cout << "selection: " << e << std::endl;
  return 0;
}
