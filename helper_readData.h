#ifndef __READ_DATA__
#define __READ_DATA__

#include "helper_progressBar.h"
#include "helper_volData/helper_readFile.h"
#include "helper_bzip.h"
#include "helper_util.h"
#include "helper_io.h"

namespace helper
{
    template<typename T>
        void readPNG(std::vector<T>& data, const std::string fname)
        {
            std::vector<uint8_t> tmp;
            readPNG(tmp, fname);
            data.resize(tmp.size());
            std::copy(tmp.begin(), tmp.end(), data.begin());
        }

    template<typename E>
        void readPNG(std::vector<V2<E>>& /*data*/, const std::string /*fname*/)
        {
            assert(false);
        }

    template<typename E>
        void readPNG(std::vector<V4<E>>& /*data*/, const std::string /*fname*/)
        {
            assert(false);
        }

    template<>
        void readPNG(std::vector<uint8_t>& data, const std::string fname)
        {
            V3<int> volDim;
            int nChannels;
            helper::cimgRead(data, volDim, nChannels, fname);
            //, bool forceGray=false, bool forceDim=false)
        }

    template<typename T, typename UD>
        size_t getTotalDataSize(const UD& ud)
        {
            return ud.getNTimeSteps() * ud.getNVoxels() * sizeof(T);
        }

    template<typename T, typename UD, typename S>
        void readData(std::vector<T>& data, const UD& ud,
                const S selectedTimeSteps)
        {
            const size_t maxNElems=std::numeric_limits<size_t>::max();
            //
            // assess data size
            //

            size_t nTimeSteps = ud.getNTimeSteps();
            const std::vector<std::string> fileNames = ud.getFileNameBuf();

            
            //std::cout << __PRETTY_FUNCTION__ << "file names " << fileNames.size() << " vs " << nTimeSteps << std::endl;
            assert(nTimeSteps == fileNames.size());

            assert(selectedTimeSteps.size() <= nTimeSteps);
            nTimeSteps = selectedTimeSteps.size();


            const size_t nVoxels = ud.getNVoxelsToRead();

            //std::cout << __PRETTY_FUNCTION__ << " there are " << nVoxels << " voxels per time step\n";

            size_t nElems = nVoxels * nTimeSteps;

            size_t deltaT = 1;
            if(nElems > maxNElems)
            {
                const size_t maxNTimeSteps = maxNElems/nVoxels;
                std::cout << "according to maxNElems (" << maxNElems << "), we can only afford " << maxNTimeSteps << " timesteps\n";
                deltaT = nTimeSteps/maxNTimeSteps;
                deltaT += (deltaT*maxNTimeSteps < nTimeSteps);
                nTimeSteps /= deltaT;
            }

            //
            // load data
            //
            data.resize(nTimeSteps);
            // std::cout << "read files"
            // 	    << " with deltaT " << deltaT
            // 	    << ", total number of time steps: " << data.size()<< "\n";
            //for(size_t i=0; i</*fileNames.size()*/data.size(); i++)

            size_t i=0;
            for(const auto t : selectedTimeSteps)
            {
#if 0
                const auto & fname=fileNames[t*deltaT];
                std::string postfix;
                switch(ud._volumeFormat)
                {
                    case volformat_raw:
                        postfix="raw";
                        break;
                    case volformat_raw_bz2:
                        postfix="bz2";
                        break;
                    case volformat_png:
                        postfix="png";
                        break;
                    default:
                        break;
                }


                std::cout << "read " << fname << std::endl;
                const bool success = helper::readFileAuto(data[i], fname, postfix);
                hassertm(success, fname);      
#else	
                switch(ud._volumeFormat)
                {
                    case volformat_raw:
                        (void)
                            helper::readRawFile(data[i], fileNames[t*deltaT], ud._volDataOffsetElems
                                    /*, nVoxels*nChannels*voxelSize*/);
                        //_dataInfo.update(_data[i]);
                        if(data[i].size() != nVoxels)
                        {
                            std::cerr << "Warning: there is mismatch between specified number of voxels (" << nVoxels<< ") and actual data elements (" << data[i].size() << "); data elements of size " << sizeof(decltype(data[0][0])) << "\n";
                        }
                        assert(data[i].size() == nVoxels);
                        break;
                    case volformat_raw_bz2:
                        bzip_decompress(data[i], fileNames[t*deltaT]);
                        break;
#if !defined(NO_CIMG_READ_DATA)
                    case volformat_png:
                        {
                            readPNG(data[i], fileNames[t*deltaT]);
                        }
                        break;
#ifdef USE_HDF5
                    case volformat_hdf5:
                        {
			  // YUYA ADDITION
			  const std::vector<std::string> HDF5GroupName = ud.getHDF5GroupName();
			  const std::vector<std::string> HDF5DatasetName = ud.getHDF5DatasetName();
			  // YUYA ADDITION END
      
                            getDataAtiHDF5(data[i], i, HDF5GroupName, HDF5DatasetName, HDF5DataDirName);
                        }
#endif
#endif
                    default:
                        std::cout << __func__ << " ERROR, UNHANDLED FORMAT "
                            << ud._volumeFormat << std::endl;
                        assert(false);
                }
#endif
                i++;
            }
            //std::cout << std::endl;
        }


    template<typename T, typename UD>
        void readData(std::vector<T>& data, const UD& ud)
        {
            readData(data, ud, helper::rangeVec(ud.getNTimeSteps()));
        }

};

#endif //__READ_DATA__
