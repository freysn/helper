#ifndef READ_FS3D_IMPL_H
#define READ_FS3D_IMPL_H

void swap2(char* buffer);
void swap4(char* buffer);
void swap8(char* buffer);
int checkIfBinary(const char *filename);
void getDataInfo(const char* filename, const int binary, 
		 int &endianness, int resolution[3], 
		 int &numberOfComponents, int &numberOfDimensions, 
		 float &time);
float* readFS3DText(const char* filename, const int resolution[3], 
		    const int numberOfComponents);
float* readFS3DBinary(const char* filename, const int resolution[3], 
		      const int numberOfComponents, const int endianness);
int getGridSpacing(const char* filename, int resolution[3],
		   float* dx, float* dy, float* dz);
float* readFS3DstagBinary(const char* filename, const int resolution[3], 
			  const int numberOfComponents, const int endianness);
#endif//READ_FS3D_IMPL_H
