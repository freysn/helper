#ifndef __LOAD_VOL_LAMMPS_DUMP_TEXT__
#define __LOAD_VOL_LAMMPS_DUMP_TEXT__

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <climits>
#include <cfloat>
#include <cmath>

#include "splitStr.h"
#include "strToNum.h"

template<typename T3, typename I3>
  void bboxMinMaxFromPGrid(T3& bboxMin, T3& bboxMax, int pId, const I3& pGrid)
{
  I3 pId3;
  pId3.z = pId/(pGrid.x*pGrid.y);
  pId3.y = (pId - pId3.z*(pGrid.x*pGrid.y))/pGrid.x;
  pId3.x = pId - pGrid.x*(pId3.y+pGrid.y*pId3.z);

  bboxMin.x = (float)pId3.x/(float)pGrid.x;
  bboxMax.x = ((float)pId3.x+1)/(float)pGrid.x;

  bboxMin.y = (float)pId3.y/(float)pGrid.y;
  bboxMax.y = ((float)pId3.y+1)/(float)pGrid.y;
  
  bboxMin.z = (float)pId3.z/(float)pGrid.z;
  bboxMax.z = ((float)pId3.z+1)/(float)pGrid.z;
}

template<typename I3>
bool pGridFromLammpsLog(I3& pGrid, std::string logFName)
{
  std::ifstream logFile (logFName.c_str());

  if(!logFile.is_open())
    {
      std::cout << "Log file " << logFName << " not found\n";
      return false;
    }

  while(!logFile.eof())
    {
      std::string line;
      std::getline (logFile,line);
      std::vector<std::string> words = split(line, ' ');
      
      if(words.empty())
        continue;
      
      /*
      std::cout << line << " " << words.size() << std::endl;
      for(size_t i=0; i<words.size(); i++)
        std::cout << "|" << words[i] << "|\n";
      */
      if(words.size() == 10 && words[8] == "processor")
        {
          pGrid.x = strToNum<float>(words[2]);
          pGrid.y = strToNum<float>(words[4]);
          pGrid.z = strToNum<float>(words[6]);
          std::cout << "determined processor grid: " << pGrid.x << " " << pGrid.y << " " << pGrid.z << std::endl;
          return true;
        }
    }

  return false;
}

template<typename I, typename T3>
  bool loadVolLammpsDumpText(std::vector<float>& vol, I& dim, std::string dumpFName, const int maxElemsPerDim, T3 bboxMin, T3 bboxMax)
{  
  std::ifstream dumpFile (dumpFName.c_str());

  if(!dumpFile.is_open())
    {
      std::cout << "Dump file " << dumpFName << " not found\n";
      return false;
    }

  dim.x = INT_MAX;
  dim.y = INT_MAX;
  dim.z = INT_MAX;

  vol.clear();

  std::string headerStr = "";

  struct bbox_t
  {
  bbox_t() :
    xmin(FLT_MAX), xmax(FLT_MAX),
      ymin(FLT_MAX), ymax(FLT_MAX),
      zmin(FLT_MAX), zmax(FLT_MAX)
    {
    }
    
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
  } bbox;

  //float cellSize = FLT_MAX;
  
  float totalxmin = FLT_MAX;
  float totalxmax = -FLT_MAX;
  float totalymin = FLT_MAX;
  float totalymax = -FLT_MAX;
  float totalzmin = FLT_MAX;
  float totalzmax = -FLT_MAX;
  
  while(!dumpFile.eof())
    {
      std::string line;
      std::getline (dumpFile,line);
      std::vector<std::string> words = split(line, ' ');
      
      if(words.empty())
        continue;
      
      if(words[0] == "ITEM:")
        {
          assert(words.size() >= 2);
          
          headerStr = words[1];
          continue;
        }
      
      if(headerStr == "BOX")
        {
          assert(words.size() >= 2);
          if(bbox.xmin == FLT_MAX)
            {
              bbox.xmin = strToNum<float>(words[0]);
              bbox.xmax = strToNum<float>(words[1]);

              std::cout << "bbox x " << bbox.xmin << " " << bbox.xmax << std::endl;
            }
          else if(bbox.ymin == FLT_MAX)
            {
              bbox.ymin = strToNum<float>(words[0]);
              bbox.ymax = strToNum<float>(words[1]);
              std::cout << "bbox y " << bbox.ymin << " " << bbox.ymax << std::endl;
            }
          else if(bbox.zmin == FLT_MAX)
            {
              bbox.zmin = strToNum<float>(words[0]);
              bbox.zmax = strToNum<float>(words[1]);\
              std::cout << "bbox z " << bbox.zmin << " " << bbox.zmax << std::endl;

              {
              assert(bbox.xmin != FLT_MAX && bbox.xmax != FLT_MAX && 
                     bbox.ymin != FLT_MAX && bbox.ymax != FLT_MAX && 
                     bbox.zmin != FLT_MAX && bbox.zmax != FLT_MAX);
              
              float maxD = std::max(bbox.xmax-bbox.xmin, 
                                    std::max(bbox.ymax-bbox.ymin,
                                             bbox.zmax-bbox.zmin));

              
              float cellSize = maxD/maxElemsPerDim;

              std::cout << cellSize << std::endl;
              dim.x = ceil((bbox.xmax-bbox.xmin)/cellSize);
              dim.y = ceil((bbox.ymax-bbox.ymin)/cellSize);
              dim.z = ceil((bbox.zmax-bbox.zmin)/cellSize);
              std::cout << "volume dimensions complete: " 
                        << dim.x << " " << dim.y << " " << dim.z << std::endl;

              
              dim.x *= (bboxMax.x-bboxMin.x);
              dim.y *= (bboxMax.y-bboxMin.y);
              dim.z *= (bboxMax.z-bboxMin.z);

              std::cout << "volume dimensions sub-division: " 
                        << dim.x << " " << dim.y << " " << dim.z << std::endl;
              
              vol.resize(dim.x*dim.y*dim.z, 0.f);
              }
            }
          else
            assert(false);
        }
            
      if(headerStr == "ATOMS")
        {          
          assert(words.size() >= 5);
          float x = strToNum<float>(words[2]);
          float y = strToNum<float>(words[3]);
          float z = strToNum<float>(words[4]);

          x = (x-bboxMin.x)/(bboxMax.x-bboxMin.x);
          y = (y-bboxMin.y)/(bboxMax.y-bboxMin.y);
          z = (z-bboxMin.z)/(bboxMax.z-bboxMin.z);

          totalxmin = std::min(x, totalxmin);
          totalymin = std::min(y, totalymin);
          totalzmin = std::min(z, totalzmin);
          
          totalxmax = std::max(x, totalxmax);
          totalymax = std::max(y, totalymax);
          totalzmax = std::max(z, totalzmax);

          /*
          int locx = x/cellSize;
          int locy = y/cellSize;
          int locz = z/cellSize;
          */
          const int locx = std::max(std::min((int)(x*dim.x), dim.x-1), 0);
          const int locy = std::max(std::min((int)(y*dim.y), dim.y-1), 0);
          const int locz = std::max(std::min((int)(z*dim.z), dim.z-1), 0);

          assert(locx >= 0 && locx < dim.x);
          assert(locy >= 0 && locy < dim.y);
          assert(locz >= 0 && locz < dim.z);

          vol[locx+dim.x*(locy+dim.y*locz)] += 1.f;          
        }
    }
  
  std::cout << "total min max: x " 
            << totalxmin << " " << totalxmax << " "
            << totalymin << " " << totalymax << " "
            << totalzmin << " " << totalzmax << "\n";

  return true;
}

#endif //__LOAD_VOL_LAMMPS_DUMP_TEXT__
