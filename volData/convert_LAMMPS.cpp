#include "helper/helper_asciiFile.h"
#include "helper/helper_string.h"
#include <vector>
#include <iostream>
#define M_VEC
#include "vec.h"
#include "helper/helper_writeFile.h"
#include "helper/helper_cmd.h"

#include "points2vol.h"
#include <limits>
#include <algorithm>

#include <random>

template<typename P>
struct BBox
{
  void operator()(const BBox<P> bbox)
  {
    (*this)(bbox.pmin);
    (*this)(bbox.pmax);
  }
  
  void operator()(const P p)
  {
    pmin = minv(pmin, p);
    pmax = maxv(pmax, p);
  }
    
  P pmin =
    P(std::numeric_limits<typename P::value_type>::max(),
      std::numeric_limits<typename P::value_type>::max(),
      std::numeric_limits<typename P::value_type>::max());
    
  P pmax =
    P(std::numeric_limits<typename P::value_type>::lowest(),
      std::numeric_limits<typename P::value_type>::lowest(),
      std::numeric_limits<typename P::value_type>::lowest());
};

template<typename P>
bool run(const std::string fname_in, int phase, BBox<P>& bbox, double& maxValue, const std::string fname_out)
{
  std::vector<std::string> sv;
  if(!helper::readASCIIv(sv, fname_in))
    {      
      return false;
    }

  enum
  {
    M_BB,
    M_ATOMS,
    M_NONE
  }
  mode = M_NONE;

  typedef typename P::value_type F;
  
  std::vector<std::pair<P, F>> points;
  std::vector<int64_t> pointIds;


  
  for(const auto e : sv)
    {
      const auto v(helper::split(e, ' '));
      if(v.size() >= 2 && v[0] == "ITEM:" && v[1] == "ATOMS")
	{
	  mode = M_ATOMS;
	  continue;
	}
      if(mode != M_ATOMS)
	continue;

      assert(v.size() >= 5);

      const P p(std::stod(v[2]), std::stod(v[3]), std::stod(v[4]));
      bbox(p);
      points.emplace_back(p, 1e-8);
      pointIds.emplace_back(std::stoi(v[0]));
    }

  std::cout << "there are " << points.size() << " points\n";

  std::cout << "bbox: " << bbox.pmin << " " << bbox.pmax << std::endl;

  if(phase == 0)
    return true;

  const auto dimf(bbox.pmax-bbox.pmin);  
  
  const int maxDim = 128;
  const auto maxDimf = maxe(dimf);
  
  if(true)
    {
      const auto nPointsOrig = points.size();

      const decltype(nPointsOrig) nNewPointsPP = 32;

      points.resize(nPointsOrig+nPointsOrig*nNewPointsPP, std::make_pair(P(0., 0., 0.), points.front().second));

      
      const double rad = 0.01*maxDimf;
      std::uniform_real_distribution<F> dis(-rad, rad);
      
      for(size_t i=0; i<nPointsOrig; i++)
	{
	  std::mt19937 gen(pointIds[i]);
	  //std::uniform_real_distribution<F> dis2(-1., 1.);	  
	  //points[i].first += P(dis2(gen), dis2(gen), dis2(gen));
	  
	  for(size_t j=0; j<nNewPointsPP; j++)
	    {
	      auto p =
		points[i].first + P(dis(gen), dis(gen), dis(gen));

	      bbox(p);
	      points[nPointsOrig + i*nNewPointsPP + j].first = p;
	    }
	}
    }

    
  if(phase == 1)
    return true;

  const F fac = (maxDim-2) / maxDimf;

  V3<int> dim(2+fac*dimf.x, 2+fac*dimf.y, 2+fac*dimf.z);

  std::cout << "fac: " << fac << " dim: " << dim << std::endl;

  for(auto & e : points)
    e.first = (e.first-bbox.pmin)*fac+P(1,1,1);

  
     

  std::vector<F> volf;
  
  splatWeightedPointsTri(volf,
			 points, dim);

  maxValue = std::max(*std::max_element(volf.begin(), volf.end()), maxValue);

  std::cout << "maxValue: " << maxValue << std::endl;

  if(phase == 2)
    return true;

  std::vector<uint8_t> vol(volf.size());

  for(size_t i=0; i<volf.size(); i++)
    vol[i] = (volf[i]/maxValue)*255.;
  
  std::cout << "write to file " << fname_out << std::endl;
  helper::writeFile(vol, fname_out);
  
  return true;
}


int main(const int argc, const char** argv)
{
  
  if(argc < 2)
    {
      std::cout << "inadequate arguments\n";
      return 1;
    }
  

  //const std::string path("/data/peri/dump_plate_2/dump/dump_plate_2_*.LAMMPS");
  //const std::string path("/homel/freysnl/dev/data/peri/dump_plate_2/dump/dump_plate_2_*.LAMMPS");
  const std::string path(std::string(argv[1]) + "/*.LAMMPS");
  //const std::string path("../data/peri/dump/dump_plate_2_*.LAMMPS");
  
  const std::vector<std::string> fnames_in(helper::cmd_ls(path));
  //const std::string fname_in(argv[1]);

  typedef double F;
  typedef V3<F> P;
  
  BBox<P> bbox;
  double maxValue = 0.;

  std::vector<std::string> fnames_out;
  for(const auto fname_in : fnames_in)
    {
      const auto s = helper::split(helper::split(fname_in, '_').back(), '.').front();
      std::cout << fname_in << ": " << s << std::endl;
      fnames_out.push_back("peri_" + helper::leadingZeros(std::stoi(s), 6));
      std::cout << fnames_out.back() << std::endl;
    }
    
  for(const auto fname_in : fnames_in)
      run(fname_in, 0, bbox, maxValue, "");

    std::cout << "-bbox: " << bbox.pmin << " " << bbox.pmax << std::endl;

    {
    auto bbox_orig = bbox;
    for(const auto fname_in : fnames_in)
      {
	auto bbox_tmp = bbox_orig;
	run(fname_in, 0, bbox_tmp, maxValue, "");
	bbox(bbox_tmp);
      }
    }

    std::cout << "-bbox: " << bbox.pmin << " " << bbox.pmax << std::endl;

    for(const auto fname_in : fnames_in)
      run(fname_in, 2, bbox, maxValue, "");

    std::cout << "-maxValue: " << maxValue << std::endl;

    for(size_t i=0; i<fnames_in.size(); i++)
      run(fnames_in[i], 3, bbox, maxValue, fnames_out[i]);
  
}
