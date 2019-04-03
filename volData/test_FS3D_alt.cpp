#include <cassert>
#include "fs3dreader_impl.h"
#include "fs3dreader_impl.cpp"
#include "../helper/helper_cmd.h"
#include "../helper/helper_string.h"
#include "../helper/helper_writeFile.h"

int main(int argc, char** argv)
{
  const std::string input = "/share/Daten/Volumen/itlmer-Dropletcollision/funs*.bin";
  const std::string output = "/share/Daten/Volumen/itlmer-Dropletcollision_raw/funs_";
  
  //std::vector<std::string> files = helper::cmd_ls(std::string(argv[1]) + "*");
  std::vector<std::string> files = helper::cmd_ls(input);

  std::sort(files.begin(), files.end());

  float minVal = std::numeric_limits<float>::max();
  float maxVal = 0.;

  minVal = 0.;
  maxVal = 1.;
  
  for(size_t pass=1; pass<2; pass++)
    for(size_t j=0; j<files.size(); j++)
      {
	const bool readDomain = true;
	int res[3];
	int numcomp;
	std::vector<double> timestepvalues;
	readDataInfo(/*_fileNameBuf.front()*/files[j], readDomain, res, 
		     numcomp, timestepvalues);

	assert(timestepvalues.size() == 1);
	std::cout << files[j] << " " << timestepvalues[0] << std::endl;
	//for(size_t i=0; i<timestepvalues.size(); i++)
	//std::cout << "timestepvalue " << i << ": " << timestepvalues[i] << std::endl;
	assert(numcomp == 1);
	//std::cout << "the data set has " << numcomp << " components\n";
	// if(numcomp == 3)
	//   _volumeDataType = voldattype_float3;

	std::cout << "volDim " << res[0] << " " << res[1] << " " << res[2] << std::endl;

	int subext[6];
	for(int d=0; d<3; d++)
	  {
	    subext[2*d] = 0;
	    subext[2*d+1] = res[d]-1;
	  }
	std::vector<float> buff;
	buff.resize(res[0]*res[1]*res[2]*numcomp);
  
	readData(files[j], res, subext, numcomp, &buff[0]);

	std::cout << "min, max: " << minVal << " " << maxVal << std::endl;
	if(pass == 0)
	  {
	    const auto minmax = std::minmax_element(buff.begin(), buff.end());
	    minVal = std::min((*minmax.first), minVal);
	    maxVal = std::max((*minmax.second), maxVal);	    
	  }
	else
	  {
	    std::vector<uint8_t>buf(buff.size());
	    for(size_t i=0; i<buff.size(); i++)
	      buf[i] = 255*((buff[i]-minVal)/(maxVal-minVal));

	    const std::string fname = output + helper::leadingZeros(j, 4);
	    std::cout << "write file " << fname << " | from " << files[j]<< std::endl;
	    helper::writeFile(buf, fname);
	  }

	
  
	// _volDim.x = res[0];
	// _volDim.y = res[1];
	// _volDim.z = res[2];

  
      }
}
