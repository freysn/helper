#include "helper_statistics.h"
#include "helper_util.h"
#include <random>

int main(int argc, char** argv)
{

  const size_t n = 200;
  
  

  std::vector<double> x(200);
  std::vector<double> w(200);

  helper::WeightedIncrementalVariance<double,double> wiv;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  
  for(size_t i=0; i<n; i++)
    {
      x[i] = dis(gen);
      w[i] = dis(gen);
      wiv.add(x[i], w[i]);
    }

  std::cout << "variance_std: " << helper::weighted_variance<double,double>(&x[0], &w[0], helper::rangeVec<>(n))<< std::endl;
  std::cout << "variance_inc: " << wiv.get() << std::endl;
    
  for(size_t i=0; i<n; i++)
    {
      x[i] = dis(gen);
      w[i] = dis(gen);
      wiv.add(x[i], w[i]);
    }

  auto indices = helper::rangeVec<>(n);
  std::shuffle(indices.begin(), indices.end(), gen);

  for(const auto i : indices)
    wiv(x[i], -w[i]);

  std::cout << "variance_inc2: " << wiv.get() << std::endl;
  
  return 0;
}
