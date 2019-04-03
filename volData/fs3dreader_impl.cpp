#include "fs3dreader_impl.h"
#include <iostream>
#include <algorithm>

static 
void swap32(unsigned& x)
{
  x = (x & 0x0000FFFF) << 16 | (x & 0xFFFF0000) >> 16;
  x = (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;  
}
static 
void swap64(unsigned long long* x)
{
  *x = ((*x) & 0x00000000FFFFFFFF) << 32 | ((*x) & 0xFFFFFFFF00000000) >> 32;
  *x = ((*x) & 0x0000FFFF0000FFFF) << 16 | ((*x) & 0xFFFF0000FFFF0000) >> 16;
  *x = ((*x) & 0x00FF00FF00FF00FF) <<  8 | ((*x) & 0xFF00FF00FF00FF00) >>  8;  
}

static 
void readComponent(std::ifstream &file, float *data, const int subext[6], 
		   const int res[3], const int numComp, const int comp)
{
  const long long int headerOffset = 188;
  const long long int compOffset = comp*res[0]*res[1]*res[2]*sizeof(double);
  const long long int subres[3] = {subext[1] - subext[0] + 1,
				   subext[3] - subext[2] + 1,
				   subext[5] - subext[4] + 1};

  double *fieldRun = new double[subres[0]];

  for (long long int k = subext[4]; k <= subext[5]; ++k) {
    for (long long int j = subext[2]; j <= subext[3]; ++j) {

      long long int dataOffset = (subext[0] + j*res[0] + k*res[0]*res[1])*sizeof(double);	  

      file.seekg(headerOffset + dataOffset + compOffset);
      file.read((char*)fieldRun, sizeof(double)*subres[0]);

      long long int runOffset = (j-subext[2])*subres[0] + (k-subext[4])*subres[0]*subres[1];
      for (int i = 0; i < subres[0]; ++i) {

      	long long int idxSubext = i + runOffset;
      	data[idxSubext*numComp + comp] = float(fieldRun[i]);
      }
    }
  }

  delete [] fieldRun;
}

void readData(std::string filename, const int res[3], const int subext[6],
	      const int numcomp, float *data)
{
  std::ifstream dataFile(filename.c_str(), std::ifstream::in | std::ifstream::binary);
  if (dataFile.good()) {
    if (numcomp == 1) { // scalar data

      readComponent(dataFile, data, subext, res, numcomp, 0);
    }
    else if (numcomp == 3) { // vector data

      readComponent(dataFile, data, subext, res, numcomp, 0);
      readComponent(dataFile, data, subext, res, numcomp, 1);
      readComponent(dataFile, data, subext, res, numcomp, 2);
    }
  }
  dataFile.close();
}

void readDataDeprecatedBinary(std::string filename, const int res[3],
			      const int subext[6], const int numcomp,
			      float *data)
{
  std::ifstream file(filename.c_str(), std::ifstream::in);
  double *dataBuffer = 0;
  bool littleEndian = true;

  if (file.good()) {

    unsigned dump;
    file.read((char*)&dump, sizeof(unsigned));

    if (dump > (1<<24)) { // Little Endian
      littleEndian = false;
    }
    file.seekg (0, file.beg);

    unsigned header, trailer;
    // Skip data information
    for (int i = 0; i < 4; i++) {

      file.read((char*)&header, sizeof(unsigned));
      if (!littleEndian) swap32(header);

      file.seekg(header, file.cur);
      file.read((char*)&trailer, sizeof(unsigned));
      if (!littleEndian) swap32(trailer);
      if (header != trailer) {
    	std::cerr << "[FS3D Reader]: Error in data file (corrupted encoding)." << std::endl;
      }
    }
    
    // Here comes actual data
    file.read((char*)&header, sizeof(unsigned));
    if (!littleEndian) swap32(header);

    dataBuffer = new double[header/sizeof(double)];

    // Read data from file to the buffer
    file.read((char*)dataBuffer, header);    

    file.read((char*)&trailer, sizeof(unsigned));
    if (!littleEndian) swap32(trailer);

    if (header != trailer) {
      std::cerr << "[FS3D Reader]: Error in data file (corrupted encoding)." << std::endl;
    }
  }
  file.close();
  //---------------------------------------------------------------------------

  int numTuples = res[0]*res[1]*res[2];
  int dataOffset = 0;  
  for (int k = 0; k < res[2]; k++) {
    for (int j = 0; j < res[1]; j++) {
      for (int i = 0; i < res[0]; i++) {

  	int bufferOffset = i + j*res[0] + k*res[0]*res[1];
	
  	for (int l = 0; l < numcomp; l++) {
	  
	  double val = dataBuffer[bufferOffset+l*numTuples];
  	  if (!littleEndian) swap64((unsigned long long*)&val);
	  
  	  data[dataOffset] = val;
  	  dataOffset++;
  	}
      }
    }
  }
  if (dataBuffer != 0) {
    delete [] dataBuffer; 
  } 
}

void readDataDeprecatedASCII(std::string filename, const int res[3],
			     const int subext[6], const int numcomp,
			     float *data)
{
  std::ifstream file(filename.c_str(), std::ifstream::in);

  if (file.good()) {

    std::string line;
    // skip the info, since we have it from the info function
    getline(file, line);
    getline(file, line);
    getline(file, line);
    getline(file, line);

    double val;
    int numTuples = res[0]*res[1]*res[2];

    for (int i = 0; i < numTuples; i++) {
      for (int c = 0; c < numcomp; c++) {	

	file >> line;
	std::replace(line.begin(), line.end(), ',', '.');
	val = atof(line.c_str());     
	int idx = i*numcomp + c;
	data[idx] = val;
      }
    }
  }
  file.close();
}

void readDataDeprecated(std::string filename, const int res[3], const int subext[6],
			const int numcomp, float *data)
{
  if (isFileBinary(filename.c_str())) {
    readDataDeprecatedBinary(filename, res, subext, numcomp, data);
  }
  else {
    readDataDeprecatedASCII(filename, res, subext, numcomp, data);
  }
}

void readCoordinates(std::string filename, const int res[3],
		     float *xcoords, float *ycoords, float *zcoords)
{
  std::fstream coordinatesFile;
  coordinatesFile.open(filename.c_str(), std::ifstream::in | std::ifstream::binary);

  if (coordinatesFile.good()) {
      
    char dump[92];
    coordinatesFile.read(dump, 92);

    double coord;
    for (int i = 0; i < res[0]; i++) {
      coordinatesFile.read((char*)&coord, sizeof(double));
      xcoords[i] = float(coord);
    }
    for (int i = 0; i < res[1]; i++) {
      coordinatesFile.read((char*)&coord, sizeof(double));
      ycoords[i] = float(coord);
    }
    for (int i = 0; i < res[2]; i++) {
      coordinatesFile.read((char*)&coord, sizeof(double));
      zcoords[i] = float(coord);
    }
  }
  coordinatesFile.close();
}

void readCoordinatesDeprecated(std::string filename, const int res[3],
			       float *xcoords, float *ycoords, float *zcoords)
{
  std::ifstream coordinatesFile(filename.c_str(), std::ifstream::in);
  if (coordinatesFile.good()) {
    
    std::string line;
    getline(coordinatesFile, line); // units
    getline(coordinatesFile, line); // resolution
    
    for (int i = 0; i < res[0]; i++) {

      double coord;
      coordinatesFile >> coord;
      xcoords[i] = coord;
    }
    for (int i = 0; i < res[1]; i++) {

      double coord;
      coordinatesFile >> coord;
      ycoords[i] = coord;
    }
    for (int i = 0; i < res[2]; i++) {

      double coord;
      coordinatesFile >> coord;
      zcoords[i] = coord;
    }
  }
  coordinatesFile.close();
}

void readDataList(std::string filename,
		  std::vector<std::string> &dataList)
{
  std::ifstream lstFile;
  lstFile.open(filename.c_str(), std::ifstream::in);

  std::string directory = std::string(filename);
  unsigned found = directory.find_last_of("/\\");
  if (found != std::string::npos) {
    directory = directory.substr(0,found+1);
  }
  else {
    directory = "";
  }

  const int lineSize = 1024;
  char line[lineSize];
  int numEntries = 0;

  while (lstFile.good()) {

    lstFile.getline(line, 1024);

    // Avoid reading empty lines
    if (std::string(line).length() > 0) {
      dataList.push_back(directory + std::string(line));
      numEntries++;
    }
  }
  lstFile.close();
}

int readDataInfo(std::string dataFileName, bool readDomain, int res[3], 
		 int &numcomp, std::vector<double> &timestepvalues)
{
  std::ifstream dataFile(dataFileName.c_str(), std::ifstream::in | std::ifstream::binary);
  if (dataFile.good()) {

    char dump[164];
    double t;
    dataFile.read(dump, 164);
    dataFile.read((char*)&t, sizeof(double));

    timestepvalues.push_back(t);

    if (readDomain) {

      int ni, nj, nk;
      int datadim;

      dataFile.read((char*)&ni, sizeof(int));
      dataFile.read((char*)&nj, sizeof(int));
      dataFile.read((char*)&nk, sizeof(int));
      dataFile.read((char*)&datadim, sizeof(int));
    
      res[0] = ni;
      res[1] = nj;
      res[2] = nk;

      if (dataFileName.find("velv") != std::string::npos) {
	numcomp = 3;
      }
      else {
	numcomp = 1;
      }
    }
  }
  else {
    return -1;
  }
  dataFile.close();
  return 0;
}

int readDataInfoDeprecatedBinary(std::string dataFileName, bool readDomain, int res[3], 
				 int &numcomp, std::vector<double> &timestepvalues)
{
  bool littleEndian = true;
 
  std::ifstream dataFile(dataFileName.c_str(), std::ifstream::in | std::ifstream::binary);
  if (dataFile.good()) {

    // check if format is little or big endian
    unsigned dump;
    dataFile.read((char*)&dump, sizeof(unsigned));
    if (dump > (1<<24)) { // Little Endian
      littleEndian = false;
      std::cerr << "BIG ENDIAN! WON'T BE HANDLED CORRECLTY!!" << std::endl;
    }
    dataFile.seekg (0, dataFile.beg);

    
    unsigned header, trailer;
    // Skip data information
    int numIter = readDomain ? 4 : 3;
    for (int i = 0; i < numIter; i++) {

      dataFile.read((char*)&header, sizeof(unsigned));
      // if (!littleEndian) swap32(header);

      if (i == 2) { // get time stamp
	dataFile.seekg(4, dataFile.cur);
	double t;	
	dataFile.read((char*)&t, sizeof(double));
	timestepvalues.push_back(t);
      }
      else if (i == 3) { // get number of components and resolution
	dataFile.read((char*)&res[0], sizeof(int));
	dataFile.read((char*)&res[1], sizeof(int));
	dataFile.read((char*)&res[2], sizeof(int));
	dataFile.read((char*)&numcomp, sizeof(int));
      }
      else {
	dataFile.seekg(header, dataFile.cur);
      }

      dataFile.read((char*)&trailer, sizeof(unsigned));
      if (!littleEndian) swap32(trailer);
      if (header != trailer) {
    	std::cerr << "[FS3D Reader]: Error in data file (corrupted encoding)." << std::endl;
      }
    }
  }
  dataFile.close();
  return 0;
}

int readDataInfoDeprecatedASCII(std::string dataFileName, bool readDomain, int res[3], 
				int &numcomp, std::vector<double> &timestepvalues)
{
  std::ifstream dataFile(dataFileName.c_str(), std::ifstream::in | std::ifstream::binary);
  if (dataFile.good()) {
    std::string line;
    double t;
    getline(dataFile, line); // description
    getline(dataFile, line); // units
    dataFile >> line >> t; // time
    timestepvalues.push_back(t);
    if (readDomain) {
      dataFile >> res[0] >> res[1] >> res[2] >> numcomp;
    }
  }
  dataFile.close();
  return 0;
}

int readDataInfoDeprecated(std::string dataFileName, bool readDomain, int res[3], 
			   int &numcomp, std::vector<double> &timestepvalues)
{
  bool binaryFile = isFileBinary(dataFileName.c_str());
  if (binaryFile) {
    readDataInfoDeprecatedBinary(dataFileName, readDomain, res, 
				 numcomp, timestepvalues);
  } else {
    readDataInfoDeprecatedASCII(dataFileName, readDomain, res, 
				numcomp, timestepvalues);
  }  
  return 0;
}

void extractSubExtent(float *coordsExt, float* coordsSub, int subext[2])
{
  for (int i = subext[0]; i <= subext[1]; ++i) {
    coordsSub[i-subext[0]] = coordsExt[i];
  }
}

bool isFileBinary(const char *filename)
{
  bool binary = false;
  std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);

  while (file.good()) {
    char c;
    file.read(&c, sizeof(char));
    if (c < 0 || (c > 13 && c < 32)) {
      binary = true;
      break;
    }
  }
  file.close();

  return binary;
}
