#define M_VEC

#include <vector>
#include "volData/vec.h"

#include "cm_viridis.h"
#include "cm_plasma.h"

#include "helper/helper_asciiFile.h"
#include <sstream>

int main(int argc, const char** argv)
{

  const auto cm = cm_viridis;
  //const auto cm = cm_plasma;

  std::vector<std::string> out;

  auto add = [&](size_t i)
    {
      std::stringstream ss;
      const double a = 0.388;
      const double b = 0.5277;
      ss << a + (b-a)*static_cast<double>(i)/(cm.size()-1) << " ";
      for(size_t j=0; j<3; j++)
	ss << (int) (255.*cm[i][j]) << " ";
      ss << 128;

      out.push_back(ss.str());
    };
  
  for(size_t i=0; i<cm.size(); i+=16)
    add(i);

  add(cm.size()-1);

  helper::writeASCIIv(out, "viridis.tff");
  //helper::writeASCIIv(out, "plasma.tff");
  
  return 0;
}
