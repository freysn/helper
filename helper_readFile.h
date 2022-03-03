#ifndef __HELPER_READ_FILE2__
#define __HELPER_READ_FILE2__

#ifndef NO_CIMG
#include "helper/helper_cimg.h"
#endif
//#include "read_FS3D_impl.h"


#include <vector>
#include <string>
#include <limits>
#include <fstream>

#include "hdf5.h"
#include "H5Cpp.h"


namespace helper
{
    void getDataAtiHDF5(std::vector<double>& data, std::string groupName, std::string datasetName, std::string dataDir)
        {

            std::cout << dataDir << std::endl;
            // Open the hdf5 file
            try{
                H5::H5File file = H5File(dataDir.c_str(), H5F_ACC_RDONLY);
                H5::Group group = file.openGroup(groupName.c_str());
                H5::DataSet dataset = group.openDataSet(datasetName.c_str());

                // Get the size
                hsize_t dims[1];
                dataset.getSpace().getSimpleExtentDims(dims, NULL);
                /* cout << "Size of data " << dims[0] << endl; */
                double* data_temp = new double[dims[0]];
                dataset.read(data_temp, H5::PredType::NATIVE_DOUBLE);
                data.resize(dims[0]);
                std::memcpy(&data[0], data_temp, sizeof(double) * dims[0]);
                dataset.close();
            } catch (Exception& e){
                std::string msg( std::string("Could not open HDF5 File\n") + e.getCDetailMsg());
                throw msg;
            }
        }
    template<typename V>
        void getDataAtiHDF5(std::vector<V>& data, std::string groupName, std::string datasetName, std::string dataDir)
        {
            std::vector<double> tmp;
            getDataAtiHDF5(tmp, groupName, datasetName, dataDir);
            data.resize(tmp.size());
            std::memcpy(&tmp[0], &data[0], tmp.size());
        }


    template<typename V>
        bool readFile2(std::vector<V>& buf, std::string volumeFileName, size_t offsetElems=0, size_t nElems=std::numeric_limits<size_t>::max())
        {
#ifdef VERBOSE
            std::cout << "Open File: " << volumeFileName << std::endl;
#endif
            std::ifstream file(volumeFileName.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

            if(!file.is_open())
            {	  
                return false;
            }

            size_t size = file.tellg();

            if(nElems == std::numeric_limits<size_t>::max())
                nElems = size/sizeof(V) - offsetElems;

            if(!((offsetElems+nElems)*sizeof(V) <= size))
                std::cerr << __func__ << " expected " << (offsetElems+nElems)*sizeof(V) << " vs data size " << size << std::endl;
            assert((offsetElems+nElems)*sizeof(V) <= size);


            file.seekg (sizeof(V)*offsetElems, std::ios::beg);
            buf.resize(nElems);
            //assert(buf.size() * sizeof(V) == size);
            file.read((char*)&buf[0], nElems*sizeof(V));

            return true;
        }  
};

#endif //__HELPER_READ_FILE__
