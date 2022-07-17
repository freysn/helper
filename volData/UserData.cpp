void init()
{
  _render = false;
  //_volumeFiles = NULL;
  _regionFileLoad = NULL;
  _regionFileSave = NULL;
  _volDim.x = _volDim.y = _volDim.z = -1;
  _voxelSize.x = _voxelSize.y = _voxelSize.z = 1.f;
  _volumeDataType = voldattype_none;
  _nTimeSteps = 0;
  _sameFileTime = true;
  _timeStepOffset = 0;
  _volDatToScalar = voldattoscalar_none;
  _volDataOffsetElems = 0;
  _volumeFormat = volformat_raw;
  _everyNthTimeStep = 1;

  _spatial_subsampling = false;

}

UserData(const std::string fname)
{
  init();
  const bool success = readConfig(fname.c_str());
  if(!success)
    {
      std::cerr << "could not read config file " << fname << std::endl;
      exit(0);
    }
}

bool readConfigOnly(const char* configFileName)
{
  std::string line;
  std::ifstream myfile (configFileName);
  if (myfile.is_open())
    {
      while (! myfile.eof() )
        {
          getline (myfile,line);
          //cout << line << endl;
          processLine((char*)line.c_str());
        }
      myfile.close();
    }
  else
    {
      printf("Error: CONFIG FILE \"%s\" NOT FOUND!\n", configFileName);
      exit(0);
      //return false;
    }
  return true;
}

bool readConfig(const char* configFileName)
{
  if(!readConfigOnly(configFileName))
    return false;

  _fileNameBuf = genFileNames();

  if(false)
    for(size_t i=0; i<_fileNameBuf.size(); i++)
      std::cout << "file " << i << ": " << _fileNameBuf[i] << std::endl;
  
  //std::cout << "volume format is " << _volumeFormat << "\n";
  if(_volumeFormat == volformat_fs3d)
    {
#if defined(NO_FS3D) || 1
      assert(false);
#else
      std::cout << "volume format is FS3D\n";
      _volumeDataType = voldattype_float;
      assert(!_fileNameBuf.empty());
      
      #if 1
      const bool readDomain = true;
      int res[3];
      int numcomp;
      std::vector<double> timestepvalues;
      readDataInfo(_fileNameBuf.front(), readDomain, res, 
		   numcomp, timestepvalues);
      //assert(numcomp == 1);
      std::cout << "the data set has " << numcomp << " components\n";
      if(numcomp == 3)
	_volumeDataType = voldattype_float3;

      _volDim.x = res[0];
      _volDim.y = res[1];
      _volDim.z = res[2];
      
      #else
      

      const int isBinary = checkIfBinary(_fileNameBuf.front().c_str());
      
      int endianness;
      int resolution[3];

      int numberOfComponents;
      int numberOfDimensions;
      float time;
  
      getDataInfo(_fileNameBuf.front().c_str(), isBinary, 
		  endianness, resolution, 
		  numberOfComponents, numberOfDimensions, 
		  time);

      std::cout << "number of components: " << numberOfComponents << std::endl;
      assert(numberOfComponents == 1);
      assert(numberOfDimensions == 3);

      _volDim.x = resolution[0];
      _volDim.y = resolution[1];
      _volDim.z = resolution[2];
      #endif
      #endif
    }

  return true;
}

void processLine(char* cline)
{
  std::string line(cline);
  //remove comments
  int index, index2;
  std::string keyword;
  index = line.find_first_of('#', 0);
  line = line.substr(0, index);
  
  index = line.find_first_of(letterString);
  
  //check if there is any statement in this line
  if(index == (int)std::string::npos)
    return;

  index2 = line.find_first_not_of(letterString, index);
  keyword = line.substr(index, index2-index);
  
  std::vector<std::string> args;
  getArguments(line.substr(index2),args);

  //cout << "Arguments for this line: \n";
  //for(int i=0; i < args.size(); i++)
  //  cout << args.at(i) << "|";
  //cout << endl;

  //std::cout << "UserData keyword " << keyword << std::endl;
  //
  // General Program settings
  //
  if(keyword == "RENDER")
    {
      if(args.at(0) == "TRUE")
        _render = true;
      else if(args.at(0) == "FALSE")
        _render = false;
      else
        {
	  std::cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
    }

  else if(keyword == "VOLUME_FILE")
    {
      _volumeFiles = args;

      for(auto &f : _volumeFiles)
	{
	  //std::cout << "volume file: |" << f << "|\n";
	  const size_t it = f.find("~");
	  if(it != std::string::npos)
	    {
	      f.replace(it, static_cast<size_t>(1), getenv("HOME"));
	    }
	}
      
      //_volumeFiles = new char*[_nVolumeFiles];
      
      /*
      for(size_t i=0; i<_nVolumeFiles;i++)
        {
          _volumeFiles[i] = new char[args.at(i).size()+1];
          strcpy(_volumeFiles[i], args.at(i).c_str());
          printf("volume file %ld: %s\n", i, _volumeFiles[i]);         
        }
      */
    }

  // YUYA ADDITION
  else if(keyword == "HDFGROUP_NAME"){
      _groupName  = args.at(0);
  }
  else if(keyword == "HDFDATASET_NAME"){
      _datasetName  = args.at(0);
  }
  // YUYA ADDITION END

  else if(keyword == "REGION_FILE_LOAD")
    {
      _regionFileLoad = new char[args.at(0).size()+1];
      strcpy(_regionFileLoad, args.at(0).c_str());
      printf("region file load: %s\n", _regionFileLoad);
    }
  else if(keyword == "REGION_FILE_SAVE")
    {
      _regionFileSave = new char[args.at(0).size()+1];
      strcpy(_regionFileSave, args.at(0).c_str());
      printf("region file save: %s\n", _regionFileSave);
    }
  else if(keyword == "VOLUME_DIM")
    {
      if(args.size() != 3)
        {
          std::cout << __func__ << " expected 3 arguments, got " << args.size() << std::endl;
          exit(0);
        }
      _volDim.x = atoi(args.at(0).c_str());
      _volDim.y = atoi(args.at(1).c_str());
      _volDim.z = atoi(args.at(2).c_str());      
    }
  else if(keyword == "VOXEL_SIZE")
    {
      _voxelSize.x = atof(args.at(0).c_str());
      _voxelSize.y = atof(args.at(1).c_str());
      _voxelSize.z = atof(args.at(2).c_str());

      std:: cout << "voxel size: " << _voxelSize.x
                 << " " << _voxelSize.y
                 << " " << _voxelSize.z
                 << std::endl;
        ;
    }
  else if(keyword == "VOLUME_DATA_TO_SCALAR")
    {
      if(args.at(0) == "X")
	_volDatToScalar = voldattoscalar_x;
      else if(args.at(0) == "Y")
	_volDatToScalar = voldattoscalar_y;
      else if(args.at(0) == "Z")
	_volDatToScalar = voldattoscalar_z;
      else if(args.at(0) == "MAG")	
	_volDatToScalar = voldattoscalar_mag;
      else
        {
	  std::cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
    }
  else if(keyword == "VOLUME_FORMAT")
    {
      if(args.at(0) == "RAW")
        _volumeFormat = volformat_raw;
      else if(args.at(0) == "RAW_XYZ")
        _volumeFormat = volformat_raw_xyz;
      else if(args.at(0) == "RAW_HURR")
        _volumeFormat = volformat_raw_hurr;
      else if(args.at(0) == "RAW_BZ2")
        _volumeFormat = volformat_raw_bz2;
      else if(args.at(0) == "PNG")
        _volumeFormat = volformat_png;
      else if(args.at(0) == "HDF5")
        _volumeFormat = volformat_hdf5;
      else if(args.at(0) == "FS3D")
	{
	  std::cout << "DETECTED volume format is FS3D\n";
	  _volumeFormat = volformat_fs3d;
	}
      else
        {
	  std::cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
    }
  else if(keyword == "VOLUME_DATA_TYPE")
    {
      if(args.at(0) == "USHORT")
        _volumeDataType = voldattype_ushort;
      else if(args.at(0) == "FLOAT")
        _volumeDataType = voldattype_float;
      else if(args.at(0) == "FLOAT2")
        _volumeDataType = voldattype_float2;
      else if(args.at(0) == "FLOAT3")
        _volumeDataType = voldattype_float3;
      else if(args.at(0) == "DOUBLE3")
        _volumeDataType = voldattype_double3;
      else if(args.at(0) == "UCHAR")
        _volumeDataType = voldattype_uchar;
      else if(args.at(0) == "DOUBLE")
        _volumeDataType = voldattype_double;
      else if(args.at(0) == "UCHAR4")
        _volumeDataType = voldattype_uchar4;
      else
        {
	  std::cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
    }
  else if(keyword == "WIDTH")
    {
      _width = atoi(args.at(0).c_str());
      printf("width: %d\n", _width);
    }
  else if(keyword == "HEIGHT")
    {
      _height = atoi(args.at(0).c_str());
      printf("height: %d\n", _height);
    }
  else if(keyword == "MAX_DIFF_FLOOD_FILL")
    {
      _maxDiffFloodFill = atof(args.at(0).c_str());
      printf("maxDiffFloodFill: %f\n", _maxDiffFloodFill);
    }
  else if(keyword == "VOLUME_DATA_OFFSET_ELEMS")
    {
      _volDataOffsetElems = atoi(args.at(0).c_str());
      std::cout << "VOL DATA ELEMS OFFSET: " << _volDataOffsetElems << std::endl;
    }
  else if(keyword == "VOLUME_TIMESTEPS_OFFSET")
    {
      _timeStepOffset = atoi(args.at(0).c_str());
      //std::cout << "TIME STEP OFFSET: " << _timeStepOffset << std::endl;
    }
  else if(keyword == "VOLUME_TIMESTEPS_EVERYNTH")
    {
      _everyNthTimeStep = atoi(args.at(0).c_str());
      std::cout << "EVERY NTH TIME STEP: " << _everyNthTimeStep << std::endl;
    }
  else if(keyword == "REGION_VOXEL_VALUE_RELATIVE_RADIUS")
    {
      _regionVoxelValueGroupRelativeRadius = atof(args.at(0).c_str());
      printf("regionVoxelValueGroupRelativeRadius: %f\n", _regionVoxelValueGroupRelativeRadius);
    }
  else if(keyword == "REGION_VOXEL_ROOT_VOXEL_CHANCE")
    {
      _regionVoxelRootVoxelChance = atof(args.at(0).c_str());
      printf("regionVoxelRootVoxelChance: %f\n", _regionVoxelRootVoxelChance);
    }
   else if(keyword == "VOLUME_TIME_FILES")
    {
      if(args.at(0) == "SAME")
        _sameFileTime = true;
      else
	{
	  _sameFileTime = false;
	  assert(args.at(0) == "COUNT");
	}
    }
    else if(keyword == "VOLUME_NUM_TIMESTEPS")
    {
      if(_nTimeSteps == 0)
	_nTimeSteps = atoi(args.at(0).c_str());
      else
	std::cout << "NUM TIME STEPS ALREADY SET TO : " << _nTimeSteps << ", IGNORE NEW INPUT OF " << atoi(args.at(0).c_str()) << std::endl;
      //assert(false);
    }

    else if(keyword == "VOLUME_NUM_FIXED_LEN")
    {
      _numFixedLen = atoi(args.at(0).c_str());
      //std::cout << "NUM FIXED LEN: " << _numFixedLen << std::endl;
      //assert(false);
    }
    // YUYA ADDITION
    else if(keyword == "TEMP_SUBSAMP_SEL")
    {
        std::fstream f(args.at(0));
        int sel;
        while (f >> sel){
            _temporal_subselection_selections.push_back(sel);
        }
        _nTimeSteps = _temporal_subselection_selections.size();
    }
  else if(keyword == "SPATIAL_SUBSAMPLING"){
      _spatial_subsampling = true;
  }
    // YUYA END

#if 0
  //VolSliceMemory NUMBER_IN_MB
  if(keyword == "VolSliceMemory")
    {
      _volSliceMemory = atof(args.at(0).c_str());
      //cout << keyword << " " << _volSliceMemory << endl;
    }

  else if(keyword == "ProjImgGraphicsMemory")
    {
      _projImgGraphicsMemory = atof(args.at(0).c_str());
      //cout << keyword << " " << _projImgGraphicsMemory << endl;
    }
  
  //TmpFilePath PATH_STRING
  else if(keyword == "TmpFilePath")
    {
      _tmpFilePath = new char[args.at(0).size()+1];
      strcpy(_tmpFilePath, args.at(0).c_str());
      //cout << keyword << " " << _tmpFilePath << endl;
    }
  //
  // Settings concerning the input projection images
  //

  //ProjImgName PATH_STRING
  else if(keyword == "ProjImgName")
    {
      _projImgsFName = new char[args.at(0).size()+1];
      strcpy(_projImgsFName, args.at(0).c_str());
      //cout << keyword << " " << _projImgsFName << endl;
    }
  else if(keyword == "ProjImgFormat")
    {
      if(args.at(0) == "FLOAT")
        _projImgType = GPUFDK_FLOAT;
      else if(args.at(0) == "SHORT")
        _projImgType = GPUFDK_SHORT;
      else if(args.at(0) == "USHORT")
        _projImgType = GPUFDK_UNSIGNED_SHORT;
      else
        {
          cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
      //cout << keyword << " " << args.at(0) << endl;
    }
  else if(keyword == "ProjImgDim")
    {
      for(int i=0; i < 3; i++)
        _projDim[i] = atoi(args.at(i).c_str());
      //cout << keyword << " " << _projDim[0] << " " << _projDim[1] << " " << _projDim[2]  << endl;
    }
  else if(keyword == "OneFilePerProjImg")
    {
      if(args.at(0) == "TRUE")
        _projImgOneFilePerImg = true;
      else if(args.at(0) == "FALSE")
        _projImgOneFilePerImg = false;
      else
        {
          cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }
      //cout << keyword << " " << _projImgOneFilePerImg << endl; 
    }
  //
  // Settings concerning the scanner
  //
  else if(keyword == "DetectorWidth")
    {
      _detectorWidth = atof(args.at(0).c_str());
      //cout << keyword << " " << _detectorWidth << endl;
    }
  else if(keyword == "DetectorHeight")
    {
      _detectorHeight = atof(args.at(0).c_str());
      //cout << keyword << " " << _detectorHeight << endl;
    }
  else if(keyword == "SourceDetectorDistance")
    {
      _sourceDetectorDistance = atof(args.at(0).c_str());
      //cout << keyword << " " << _sourceDetectorDistance << endl;
      _sourceOriginDistance = _sourceDetectorDistance - _detectorObjectDistance;
    }
  else if(keyword == "DetectorObjectDistance")
    {
      _detectorObjectDistance = atof(args.at(0).c_str());
      //cout << keyword << " " << _detectorObjectDistance << endl;
      _sourceOriginDistance = _sourceDetectorDistance - _detectorObjectDistance;
    }
  
  //
  // Settings concerning the output volume
  //
  else if(keyword == "OutVolDim")
    {
      for(int i=0; i < 3; i++)
        _volDim[i] = atoi(args.at(i).c_str());
      //cout << keyword << " " << _volDim[0] << " " << _volDim[1] << " " << _volDim[2]  << endl;
    }
  else if(keyword == "OutVolFileName")
    {
      _outputVolumeFName = new char[args.at(0).size() + 1];
      strcpy(_outputVolumeFName, args.at(0).c_str());
      //cout << keyword << " " << _outputVolumeFName << endl;
    }
  else if(keyword == "Prefiltered")
    {
      if(args.at(0) == "TRUE")
        _prefiltered = true;
      else if(args.at(0) == "FALSE")
        _prefiltered = false;
      else
        {
          cerr << args.at(0) << " is no valid option\n";
          exit(0);
        }  
    }

  
  /*
  else if(keyword == "")
    {
    }
  */
#endif
  else
    {
      std::cerr << "Parse Error: " << keyword << " is not a valid keyword\n";
      exit(0);
    }

  

}
