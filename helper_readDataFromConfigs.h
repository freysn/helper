#ifndef __HELPER_READ_DATA_FROM_CONFIG__
#define __HELPER_READ_DATA_FROM_CONFIG__

#include "helper_readData.h"
#include "helper_volData/UserData.h"

namespace helper
{

using UserData_t = UserData<V3<int>,V3<double>>;



template<typename elem_t, typename S=std::vector<size_t>>
std::tuple<std::vector<std::vector<elem_t>>,V4<size_t>> readDataFromConfig(const std::string configFile, const S selectedTimeSteps = S())
{
  UserData_t ud;

  std::vector<std::vector<elem_t>> tmp;

  V4<size_t> dim(-1, -1, -1, -1);
	  
  {    
    const bool success = ud.readConfigOnly(configFile.c_str());
    hassert(success);

#if __APPLE__
    {
      const std::string zydecoBasePath("/media/hdd_data");
      if(ud._volumeFiles[0].find(zydecoBasePath) != std::string::npos)
    {
      ud._volumeFiles[0].replace(0, std::string(zydecoBasePath).length(), "/Users/freysn/dev");
      std::cout << "|NEW FILE " << ud._volumeFiles[0]<< "|\n" ;
    }
    }
#endif
    ud._fileNameBuf = ud.genFileNames();
    const std::tuple<size_t, size_t, size_t, size_t> dataType = ud.getType();

    dim.x = ud._volDim.x;
    dim.y = ud._volDim.y;
    dim.z = ud._volDim.z;
    dim.w = 0;
    
    // const bool is_f32 = (std::get<0>(dataType) == typeid(float).hash_code());
    // const bool is_f64 = (std::get<0>(dataType) == typeid(double).hash_code());
    // const bool is_u8 = (std::get<0>(dataType) == typeid(uint8_t).hash_code());
    
    auto readWrapper = 
      [&selectedTimeSteps, &ud](auto & tmp)
      {
	if(selectedTimeSteps.empty())
	  helper::readData(tmp, ud);
	else
	  helper::readData(tmp, ud, selectedTimeSteps);
      };
    
    // auto readWrapperConvert =
    //   [readWrapper, &tmp](auto & tmp2)
    //   {
    // 	readWrapper(tmp2);
    // 	tmp.resize(tmp2.size());
    // 	for(size_t i=0; i<tmp.size(); i++)
    // 	  {
    // 	    tmp[i].resize(tmp2[i].size());
    // 	    for(size_t j=0; j<tmp[i].size(); j++)
    // 	      assign(tmp[i][j],tmp2[i][j]);
    // 	    //std::copy(tmp2[i].begin(), tmp2[i].end(), tmp[i].begin(), []);
    // 	  }
    //   };

#if 0
    if(std::get<0>(dataType) == typeid(V2<float>).hash_code())
      {
	std::vector<std::vector<V2<float>>> data;
	if(selectedTimeSteps.empty())
	  helper::readData(data, ud);
	else
	  helper::readData(data, ud, selectedTimeSteps);
	assert(success);
	tmp.resize(data.size());

	assert(dim.w == 0 || data.size() == dim.w);

	dim.w = data.size();
	
	for(size_t t=0; t<data.size(); t++)
	  {
	    assert(data[t].size() == dim.x*dim.y*dim.z);
	    tmp[t].resize(data[t].size());
	    for(size_t v=0; v<data[t].size(); v++)
	      tmp[t][v] = length(data[t][v]);
	  }		      
      }
    else
#endif
      if(std::get<0>(dataType) == typeid(elem_t).hash_code())
      {
	readWrapper(tmp);
      }
#if 0
    else if(is_f32)
      {
	std::vector<std::vector<float>> tmp2;
	readWrapperConvert(tmp2);
      }
    else if(is_f64)
      {
	std::vector<std::vector<double>> tmp2;
	readWrapperConvert(tmp2);
      }
    else if(is_u8)
      {
	std::vector<std::vector<uint8_t>> tmp2;
	readWrapperConvert(tmp2);
      }
#endif
    else
      throw "data type not captured";
      
  }
  
  return std::make_tuple(tmp, dim);
}

template<typename elem_t, typename S>
auto readDataFromConfigs(const std::vector<std::string> files, const S selectedTimeSteps)
{
  UserData_t ud;

  std::vector<std::vector<std::vector<elem_t>>> dataVec;


  V4<size_t> dim(-1, -1, -1, -1);
	  

  dataVec.resize(files.size());
  
  for(size_t i=0; i<dataVec.size(); i++)
  {
    std::tie(dataVec[i], dim) = readDataFromConfig<elem_t>(files[i], selectedTimeSteps);
  }  

  return std::make_tuple(dataVec, dim);
}
}

#endif //__READ_DATA_FROM_CONFIG__
