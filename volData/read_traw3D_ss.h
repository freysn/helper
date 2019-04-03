#ifndef __READ_TRAW3D_SS__
#define __READ_TRAW3D_SS__

#include <vector>

bool read_traw3D_ss(std::vector<short>& volume,
                    int3& volDim, float3& voxelSize, const std::string& fname)
{
  // "*.traw3D_ss"
  FILE *fp = fopen(fname.c_str(), "rb");
  if( fp == 0 ) return false;

  int    W , H, D; // resoulution (Width, Height, Depth)
  fread( &W , sizeof(int   ), 1, fp ); 
  fread( &H , sizeof(int   ), 1, fp ); 
  fread( &D , sizeof(int   ), 1, fp );
  volDim.x = W;
  volDim.y = H;
  volDim.z = D;

  double px,py,pz; // pitch in x,y,z axes
  fread( &px, sizeof(double), 1, fp ); 
  fread( &py, sizeof(double), 1, fp ); 
  fread( &pz, sizeof(double), 1, fp );
  voxelSize.x = px;
  voxelSize.y = py;
  voxelSize.z = pz;

  //read signed short array
  //short *volume = new short[ W*H*D ];
  volume.resize(W*H*D);
  if( fread( &volume[0], sizeof(short), W*H*D, fp ) != W*H*D ) { fclose( fp ); return false;}

  fclose(fp);
  return true;
}

#endif //__READ_TRAW3D_SS__
