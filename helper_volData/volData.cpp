#include "volData.h"
#include <climits>
#include <cfloat>

mm_uint3 resizeDim(mm_uint3 from, size_t nMaxElements)
{  
  mm_uint total = from.x + from.y + from.z;
  double ratioX = (double) from.x / (double) total;
  double ratioY = (double) from.y / (double) total;
  double ratioZ = (double) from.z / (double) total;

  mm_uint3 to = from;
  
  total = to.x*to.y*to.z;
  while(total > nMaxElements)
    {
      double x = (double)to.x/(ratioX*total);
      double y = (double)to.y/(ratioY*total);
      double z = (double)to.z/(ratioZ*total);

      if(x > y)
	{
	  if(x > z)
	    {
	      to.x--;
	    }
	  else
	    {
	      to.z--;
	    }
	}
      else
	{
	  if(y>z)
	    {
	      to.y--;
	    }
	  else
	    {
	      to.z--;
	    }
	}
      total = to.x*to.y*to.z;
    }
  
  return to;
}



mm_ushort *loadRawFileUShort(char *filename, size_t size, size_t offset)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    exit(-1);
    return 0;
  }
  fseek (fp , offset , SEEK_SET );
  mm_ushort* data = new mm_ushort[size];
  size_t read = fread(data, sizeof(mm_ushort), size, fp);

  fclose(fp);
  
  //convert from 12 to 16 bit data
  //for(int i=0; i < size; i++)
  //  data[i] *= 16;
  
  printf("Read '%s', %ld bytes (given %ld bytes)\n", filename, read, size);
  
  return data;
}

// Load raw data from disk
unsigned char *loadRawFileUCharPlain(char *filename, size_t size, size_t offset)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    exit(-1);
    return 0;
  }
  fseek (fp , offset , SEEK_SET );
  unsigned char* data = new mm_uchar[size];
  size_t read = fread(data, sizeof(mm_uchar), size, fp);
      
  fclose(fp);
      
  printf("Read '%s', %ld bytes  (given %ld bytes)\n", filename, read, size);
  
  return data;
}

// Load raw data from disk
mm_ushort *loadRawFileUChar(char *filename, size_t size, size_t offset)
{
  
  mm_uchar *tmpData = loadRawFileUCharPlain(filename, size);
    
  mm_ushort* data = new mm_ushort[size];
  for(int i=0; i < size; i++)
    data[i] = ((unsigned short) tmpData[i]) << 8;
  
  delete [] tmpData;
  
  //printf("Read '%s', %ld bytes  (given %ld bytes)\n", filename, read, size);
  
  return data;
}

// Load raw data from disk
mm_ushort *loadRawFileFloat(char *filename, size_t size, size_t offset)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    exit(-1);
    return 0;
  }
  fseek (fp , offset , SEEK_SET );
  mm_ushort *data = new mm_ushort[size];
  float* tmpData = new float[size];
  
  size_t read = fread(tmpData, sizeof(float), size, fp);
  fclose(fp);
  
  float biggestValue = FLT_MIN;
  float smallestValue = FLT_MAX;
  for(int i=0; i < size;i++)
    {
      smallestValue = std::min(smallestValue, tmpData[i]);
      biggestValue = std::max(biggestValue, tmpData[i]);
    }
  
  printf("smallestValue: %f\n", smallestValue);
  printf("biggestValue: %f\n", biggestValue);

  for(int i=0; i < size;i++)
    data[i] = 
      (mm_ushort) 
      (((float)0xffff)* 
       (tmpData[i]-smallestValue)/(biggestValue-smallestValue));
    
  printf("Read '%s', %ld bytes\n", filename, read);
  delete [] tmpData;
  
  return data;
}




mm_ushort* loadRawFile(char *filename, size_t size, elemType_t elemType, size_t offset)
{
  mm_ushort* data = 0;
  switch(elemType)
    {
    case UCHAR:
      data = loadRawFileUChar(filename, size, offset);
      break;
    case USHORT:
      data = loadRawFileUShort(filename, size, offset);
      break;
    case FLOAT:
      data = loadRawFileFloat(filename, size, offset);
      break;
    default:
      assert(false);
    }

  return data;
}

