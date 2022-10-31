#include "helper_SOM.h"

int main(int argc, char** argv)
{

  std::mt19937 rng = std::mt19937(18);
  std::uniform_real_distribution<double> dis  = std::uniform_real_distribution<double>(0., 1.);

  using SOM_t = helper::helper_SOM_norm_ring<3,8,double>;
  SOM_t som;

  using vec_t=SOM_t::vec_t;

  std::vector<vec_t> trainingData(2048);

  for(auto& v : trainingData)
    for(auto& e : v)
      e = dis(rng);

  std::cout << "trainingDataSize: " << trainingData.size() << std::endl;

  for(uint64_t pass=0; pass<32; pass++)
    {
      std::cout << som << std::endl;
      for(const auto v : trainingData)      
	som.train(v);
      
      som.moveOn();
    }
    
  std::cout << som << std::endl;
  return 0;
}
