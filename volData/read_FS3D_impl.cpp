#include <cstdio>
#include <cstdlib>
#include <cctype>
#include "read_FS3D_impl.h"

#define LINE_SIZE 1024

//==============================================================================
void swap2(char* buffer)
{
  char c;
  
  c = buffer[0];
  buffer[0] = buffer[1];
  buffer[1] = c;
}
//==============================================================================
void swap4(char* buffer)
{
  char c;
  
  c = buffer[0];
  buffer[0] = buffer[3];
  buffer[3] = c;

  c = buffer[1];
  buffer[1] = buffer[2];
  buffer[2] = c;
}
//==============================================================================
void swap8(char* buffer)
{
  char c;

  c = buffer[0];
  buffer[0] = buffer[7];
  buffer[7] = c;

  c = buffer[1];
  buffer[1] = buffer[6];
  buffer[6] = c;

  c = buffer[2];
  buffer[2] = buffer[5];
  buffer[5] = c;

  c = buffer[3];
  buffer[3] = buffer[4];
  buffer[4] = c;  
}
//==============================================================================
int checkIfBinary(const char *filename)
{
  FILE *f;
  int fSize;
  char *buffer;
  size_t bytesCopied;
  int i;
  int binary;

  f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
  }

  fseek(f, 0, SEEK_END);
  fSize = ftell(f);
  rewind(f);

  buffer = (char*)malloc(fSize*sizeof(char));
  if (buffer == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not allocate memory.\n");
  }

  bytesCopied = fread(buffer, 1, fSize, f);
  if (bytesCopied != fSize) {
    fprintf(stderr, "[FS3D Reader]: Reading error.\n");
  }

  for (i = 0; i < fSize; i++) {
    if (buffer[i] < 0 || (buffer[i] > 13 && buffer[i] < 32))
      break;
  }

  fclose(f);
  free(buffer);

  if (i == fSize)
    binary = 0;
  else
    binary = 1;

  return binary;
}
//==============================================================================
void getDataInfo(const char* filename, const int binary, 
		 int &endianness, int resolution[3], 
		 int &numberOfComponents, int &numberOfDimensions, 
		 float &time)
{
  if (binary) {
    FILE *f;
    char line[LINE_SIZE];
    unsigned int header, trailer;
    int fSize;
    char *buffer;
    unsigned int dump;
    double timed;

    f = fopen(filename, "rb");
    if (f == NULL) {
      fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
    }

    fseek(f, 0, SEEK_END);
    fSize = ftell(f);
    rewind(f);
      
    buffer = (char*)malloc(fSize*sizeof(char));

    // Check Endianness
    fread(&dump, sizeof(unsigned int), 1, f);

    if (dump < 256*256*256) // Hack!!
      endianness = 0; // Little Endian
    else
      endianness = 1; // Big Endian

    rewind(f);
    //-----------------

    // description (string)
    fread(&header, sizeof(unsigned int), 1, f);
    if (endianness) swap4((char*)&header);
    fread(buffer, 1, header, f);
    fread(&trailer, sizeof(unsigned int), 1, f);
    if (endianness) swap4((char*)&trailer);
      
    if (header != trailer)
      fprintf(stderr, 
	      "[FS3D Reader]: Error in data file (corrupted encoding).");

    // units (string)
    fread(&header, sizeof(unsigned int), 1, f);
    if (endianness) swap4((char*)&header);
    fread(buffer, 1, header, f);
    fread(&trailer, sizeof(unsigned int), 1, f);
    if (endianness) swap4((char*)&trailer);

    if (header != trailer)
      fprintf(stderr, 
	      "[FS3D Reader]: Error in data file (corrupted encoding).");
      
    // cycle, time (unsigned int, double)
    fread(&header, sizeof(unsigned int), 1, f);
    fread(&dump, sizeof(unsigned int), 1, f);
    fread(&timed, sizeof(double), 1, f);
    fread(&trailer, sizeof(unsigned int), 1, f);
    if (header != trailer)
      fprintf(stderr, 
	      "[FS3D Reader]: Error in data file (corrupted encoding).");

    // resolution, number of components
    fread(&header, sizeof(unsigned int), 1, f);
    fread(resolution, 3*sizeof(int), 1, f);
    fread(&numberOfComponents, sizeof(int), 1, f);
    fread(&trailer, sizeof(unsigned int), 1, f);
    if (header != trailer)
      fprintf(stderr, 
	      "[FS3D Reader]: Error in data file (corrupted encoding).");

    // Swap bytes if necessary
    if (endianness) swap8((char*)&timed);
    time = (float)timed;

    if (endianness) swap4((char*)&resolution[0]);
    if (endianness) swap4((char*)&resolution[1]);
    if (endianness) swap4((char*)&resolution[2]);
    if (endianness) swap4((char*)&numberOfComponents);

    numberOfDimensions = 3;

    free(buffer);
    fclose(f);
  }
  else {
    FILE *f;
    char line[LINE_SIZE];
    unsigned int dump;

    f = fopen(filename, "r");
    if (f == NULL) {
      fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
    }
      
    // Get data information
    fgets(line, LINE_SIZE, f);
    fgets(line, LINE_SIZE, f);
    fgets(line, LINE_SIZE, f);
    sscanf(line, "%u %f", &dump, &time);
    fgets(line, LINE_SIZE, f);
    sscanf(line, "%d %d %d %d", &(resolution[0]), &(resolution[1]), 
	   &(resolution[2]), &(numberOfComponents));
      
    // Assuming 3-dimensional data
    numberOfDimensions = 3;
      
    fclose(f);
  }
}
//==============================================================================
int testComma()
{
  int comma;
  char floatingPointNumberString[] = "0.562E-03";
  float floatingPointNumber;

  sscanf(floatingPointNumberString, "%f", &floatingPointNumber);

  if (floatingPointNumber == 0.0f)
    comma = 1;
  else
    comma = 0;

  return comma;
}
//==============================================================================
void replaceChar(char* line, char c0, char c1)
{
  int i = 0;

  while (line[i] != '\n') {
    if (line[i] == c0)
      line[i] = c1;
    i++;
  }
}
//==============================================================================
float* readFS3DText(const char* filename, const int resolution[3], 
		    const int numberOfComponents)
{
  int numFloats = resolution[0]*resolution[1]*resolution[2]*numberOfComponents;
  float *data = NULL;
  FILE *f;
  char line[LINE_SIZE];
  int i, j, k, l, n;
  int extent[6];
  int dataOffset = 0;
  float *dataBuffer = NULL;

  //-----------------------------------------------------------------------------
  // Read data from file
  f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
  }

  // Skip data information
  fgets(line, LINE_SIZE, f);
  fgets(line, LINE_SIZE, f);
  fgets(line, LINE_SIZE, f);
  fgets(line, LINE_SIZE, f);

  // Load data------------------------------------------------------------------
  dataBuffer = (float*)malloc(numFloats*sizeof(float));
  i = 0;

  const int floatingComma = testComma();

  while (fgets(line, LINE_SIZE, f) != NULL)
    {
      if (floatingComma)
      	replaceChar(line, '.', ',');
      if (numberOfComponents == 1)
	sscanf(line, "%f", &(dataBuffer[i++]));
      else if (numberOfComponents == 3)
	sscanf(line, "%f %f %f", &(dataBuffer[i++]), &(dataBuffer[i++]), 
	       &(dataBuffer[i++]));
    }

  if (i != numFloats)
    fprintf(stderr, "[FS3D Reader]: Error while loading data. " \
		  "Filling remaining elements with zeros\n");
  for (; i < numFloats; i++)
    {
      dataBuffer[i] = 0.0f;
    }
  
  fclose(f);

  //-----------------------------------------------------------------------------
  n = numberOfComponents;
  dataOffset = 0;

  data = (float*)malloc(numFloats*sizeof(float));

  for (k = 0; k < resolution[2]; k++)
    for (j = 0; j < resolution[1]; j++)
      for (i = 0; i < resolution[0]; i++) {

	int bufferOffset = i + j*resolution[0] + k*resolution[0]*resolution[1];

	for (l = 0; l < n; l++) {
	  data[dataOffset] = dataBuffer[bufferOffset*n+n-l-1];
	  dataOffset++;
	}
      }
  free(dataBuffer);

  return data;
}
//==============================================================================
float* readFS3DBinary(const char* filename, const int resolution[3], 
		      const int numberOfComponents, const int endianness)
{
  int numFloats = resolution[0]*resolution[1]*resolution[2]*numberOfComponents;
  int numTuples = resolution[0]*resolution[1]*resolution[2];
  unsigned int header, trailer;
  double *dataBuffer = NULL;
  float *data = NULL;
  int i, j, k, l, n;
  int dataOffset;
  int extent[6];
  FILE *f;

  f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
  }

  // Skip data information
  for (int i = 0; i < 4; i++)
    {
      fread(&header, sizeof(unsigned int), 1, f);
      if (endianness) swap4((char*)&header);
      fseek(f, header, SEEK_CUR);
      fread(&trailer, sizeof(unsigned int), 1, f);
      if (endianness) swap4((char*)&trailer);
      if (header != trailer)
	fprintf(stderr, 
		"[FS3D Reader]: Error in data file (corrupted encoding).");
    }

  // Here comes actual data
  fread(&header, sizeof(unsigned int), 1, f);
  if (endianness) swap4((char*)&header);
  dataBuffer = new double[header/sizeof(double)];
  // Read data from file to the buffer
  fread(dataBuffer, header, 1, f);
  fread(&trailer, sizeof(unsigned int), 1, f);
  if (endianness) swap4((char*)&trailer);
  if (header != trailer)
    fprintf(stderr, "[FS3D Reader]: Error in data file (corrupted encoding).");

  fclose(f);
  //-----------------------------------------------------------------------------
  n = numberOfComponents;
  dataOffset = 0;

  data = (float*)malloc(numFloats*sizeof(float));

  for (k = 0; k < resolution[2]; k++)
    for (j = 0; j < resolution[1]; j++)
      for (i = 0; i < resolution[0]; i++) {
	int bufferOffset = i + j*resolution[0] + 
	  k*resolution[0]*resolution[1];
	
	for (l = 0; l < n; l++) {
	  if (endianness) swap8((char*)&dataBuffer[bufferOffset+l*numTuples]);
	  data[dataOffset] =  dataBuffer[bufferOffset+l*numTuples];
	  dataOffset++;
	}
      }
  free(dataBuffer);

  return data;
}
//==============================================================================
int getGridSpacing(const char* filename, int resolution[3],
		   float* dx, float* dy, float* dz)
{
  FILE *f = fopen(filename, "r");
  char line[LINE_SIZE];
  int lres[3];

  if (f == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not open file %s.\n", filename);
    fprintf(stderr, "[FS3D Reader]: Setting spacing to 1.0.\n");
    for (int idx = 0; idx < resolution[0]; idx++)
      dx[idx] = idx;
    for (int idx = 0; idx < resolution[1]; idx++)
      dy[idx] = idx;
    for (int idx = 0; idx < resolution[2]; idx++)
      dz[idx] = idx;
    
    return -1;
  }

  fgets(line, LINE_SIZE, f);
  fgets(line, LINE_SIZE, f);
  sscanf(line, "%d %d %d", &lres[0], &lres[1], &lres[2]);

  if (lres[0] != resolution[0] || 
      lres[1] != resolution[1] || 
      lres[2] != resolution[2]) {
    fprintf(stderr, 
	    "Resolution in the file %s and in the loaded data don't match!\n");
    return -1;
  }

  const int floatingComma = testComma();

  for (int idx = 0; idx < lres[0]; idx++) {
    fgets(line, LINE_SIZE, f);
    if (floatingComma)
      replaceChar(line, '.', ',');
    sscanf(line, "%f", &dx[idx]);
  }
  for (int idx = 0; idx < lres[1]; idx++) {
    fgets(line, LINE_SIZE, f);
    if (floatingComma)
      replaceChar(line, '.', ',');
    sscanf(line, "%f", &dy[idx]);
  }
  for (int idx = 0; idx < lres[2]; idx++) {
    fgets(line, LINE_SIZE, f);
    if (floatingComma)
      replaceChar(line, '.', ',');
    sscanf(line, "%f", &dz[idx]);
  }
  return 0;
}
//==============================================================================
float* readFS3DstagBinary(const char* filename, const int resolution[3], 
			  const int numberOfComponents, const int endianness)
{
  int numXFloats = (resolution[0]+1)*resolution[1]*resolution[2];
  int numYFloats = resolution[0]*(resolution[1]+1)*resolution[2];
  int numZFloats = resolution[0]*resolution[1]*(resolution[2]+1);
  int numFloats = numXFloats + numYFloats + numZFloats;
  
  unsigned int header, trailer;
  double *dataBuffer = NULL;
  float *data = NULL;
  int i, j, k, l, n;
  int dataOffset;
  int extent[6];
  FILE *f;

  f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[FS3D Reader]: Could not read file %s.\n", filename);
  }

  // Skip data information
  for (int i = 0; i < 4; i++)
    {
      fread(&header, sizeof(unsigned int), 1, f);
      if (endianness) swap4((char*)&header);
      fseek(f, header, SEEK_CUR);
      fread(&trailer, sizeof(unsigned int), 1, f);
      if (endianness) swap4((char*)&trailer);
      if (header != trailer)
	fprintf(stderr, 
		"[FS3D Reader]: Error in data file (corrupted encoding).");
    }

  // Here comes actual data
  fread(&header, sizeof(unsigned int), 1, f);
  if (endianness) swap4((char*)&header);
  dataBuffer = new double[header/sizeof(double)];
  // Read data from file to the buffer
  fread(dataBuffer, header, 1, f);
  fread(&trailer, sizeof(unsigned int), 1, f);
  if (endianness) swap4((char*)&trailer);
  if (header != trailer)
    fprintf(stderr, "[FS3D Reader]: Error in data file (corrupted encoding).");

  fclose(f);
  //-----------------------------------------------------------------------------

  data = (float*)malloc(numFloats*sizeof(float));
  
  for (i = 0; i < numXFloats; i++) {
    if (endianness) swap8((char*)&dataBuffer[i]);
    data[i] = dataBuffer[i];
  }
  for (j = 0; j < numYFloats; j++) {
    if (endianness) swap8((char*)&dataBuffer[j]);
    data[numXFloats+j] = dataBuffer[j];
  }
  for (k = 0; k < numZFloats; k++) {
    if (endianness) swap8((char*)&dataBuffer[k]);
    data[numXFloats+numYFloats+k] = dataBuffer[k];
  }

  free(dataBuffer);

  return data;
}
//==============================================================================
#undef LINE_SIZE
