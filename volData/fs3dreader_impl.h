#ifndef FS3DREADER_IMPL_H
#define FS3DREADER_IMPL_H

#include <string>
#include <vector>
#include <fstream>

void readData(std::string filename, const int res[3], const int subext[6],
	      const int numcomp, float *data);

void readDataDeprecated(std::string filename, const int res[3], const int subext[6],
			const int numcomp, float *data);

void readCoordinates(std::string filename, const int res[3],
		     float *xcoords, float *ycoords, float *zcoords);
void readCoordinatesDeprecated(std::string filename, const int res[3],
			       float *xcoords, float *ycoords, float *zcoords);

void readDataList(std::string filename,
		  std::vector<std::string> &dataList);

int readDataInfo(std::string dataFileName, bool readDomain, int res[3], 
		 int &numcomp, std::vector<double> &timestepvalues);

int readDataInfoDeprecated(std::string dataFileName, bool readDomain, int res[3], 
			   int &numcomp, std::vector<double> &timestepvalues);

void extractSubExtent(float *coordsExt, float* coordsSub, int subext[2]);

bool isFileBinary(const char *filename);

#endif//FS3DREADER_IMPL_H
