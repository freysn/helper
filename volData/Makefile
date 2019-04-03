OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

cxx ?= g++

HOST=$(shell hostname)
ifeq "$(HOST)" "hestia1"
CXX=/home/freysn/dev/ext/gcc-5.3.0_build/bin/g++
else
CXX=${cxx}
endif



ifeq ($(dbg),1)
CXX_OPT_DBG_FLAGS = -g
else
CXX_OPT_DBG_FLAGS = -O3
endif

ifneq ($(DARWIN),)
#CXX_CMD := clang++ -stdlib=libc++
stdlib ?= libc++
ifeq ($(stdlib),libstdc++)
CXX_OPT_DBG_FLAGS += -DNO_CPP11
endif
#stdlib=libstdc++
#CXX_CMD := clang++ -stdlib=libstdc++
#CXX_CMD := clang++ -stdlib=$(stdlib)
#CXX_CMD = clang-mp-3.9 -std=c++11
CXX_CMD = clang-mp-5.0 -std=c++14 -lc++
#CXX_CMD = g++-6

#CXX_CMD := $(CXX) -std=c++11
else
CXX_CMD := $(CXX) -std=c++14
#CXX_CMD = /home/freysn/tmp/build/build/bin/clang -std=c++11
endif

CXX = $(CXX_CMD) $(CXX_OPT_DBG_FLAGS) -DNO_FS3D

verbose ?= 0

ifeq ($(verbose),1)
CXX += -DVERBOSE
endif

CXX_FLAGS = -Wall -Wno-deprecated -Wno-sign-compare 
#-fopenmp

ifeq ($(DARWIN),)
CXX_FLAGS += -DUSE_OMP -fopenmp
endif

#LIB_TARGET = libVolData.so

ifneq ($(DARWIN),)
LIB_TARGET = libVolData_$(stdlib).a
else
LIB_TARGET = libVolData.a
endif

INCLUDES += -I.. -I/opt/local/include

CCFILES = volData.cpp UserData.cpp fs3dreader_impl.cpp

#read_FS3D_impl.cpp

#OBJS :=  $(patsubst %.cpp, %.o,$(notdir $(CCFILES)))
OBJS :=  $(patsubst %.cpp, %.o,$(CCFILES))

#$(CXX) -shared $(CXX_FLAGS) -o $(LIB_TARGET) $(LIB) $(OBJS)

.PHONY: convert vdb clip convert_LAMMPS

#all: lib
all: convert_LAMMPS

convert_LAMMPS:
	$(CXX_CMD) convert_LAMMPS.cpp -o convert_LAMMPS -I..
lib: $(OBJS)
	ar rcs $(LIB_TARGET) $(LIB) $(OBJS)

%.o : %.cpp
	$(CXX) -fPIC $(CXX_FLAGS) $(INCLUDES) -I$(CUDA_INC) -I$(CUDA_SDK_INC) -o $@ -c $<

transform:
	$(CXX) -std=c++11 testTransform.cpp $(CXX_FLAGS) $(INCLUDES) -o testTransform -lpng -lpthread $(LIB_TARGET) -I/opt/local/include -L/opt/local/lib -I..

vdb:
	$(CXX) test_openvdb.cpp -I/opt2/local/include -I/opt/local/include -L/opt/local/lib -ltbb -ltbbmalloc -ltbbmalloc_proxy -L/opt2/local/lib -lopenvdb -lIlmImf -lIlmThread -lImath -lIex -lImlib2 -lHalf -I/opt2/build $(LIB_TARGET) -o vdb

clip: clip.cpp
	$(CXX) -std=c++11 clip.cpp $(CXX_FLAGS) $(INCLUDES) -o clip $(LIB_TARGET) -lpng -lpthread -lbz2 -L/opt/local/lib

convert: convert.cpp
	$(CXX) -std=c++11 convert.cpp $(CXX_FLAGS) $(INCLUDES) -o convert $(LIB_TARGET) -lpng -lpthread

convert_traw3D: convert_traw3D.cpp read_traw3D_ss.h
	$(CXX) convert_traw3D.cpp $(CXX_FLAGS) $(INCLUDES) -o convert_traw3D

fs3d:$(LIB_TARGET) test_FS3D.cpp
	$(CXX) test_FS3D.cpp $(CXX_FLAGS) $(INCLUDES) $(LIB_TARGET) -o test_FS3D
convVolData: $(LIB_TARGET) convVolData.cpp
	$(CXX) convVolData.cpp $(CXX_FLAGS) $(INCLUDES) $(LIB_TARGET) -o convVolData

convVolData2: convVolData2.cpp
	$(CXX) convVolData2.cpp $(CXX_FLAGS) $(INCLUDES) -o convVolData2

createTestVolData: createTestVolData.cpp
	$(CXX) -std=c++11 createTestVolData.cpp $(CXX_FLAGS) $(INCLUDES) -o createTestVolData

createCuboid: createCuboid.cpp
	$(CXX) -std=c++11 createCuboid.cpp $(CXX_FLAGS) $(INCLUDES) -o createCuboid

shepard: testShepard.cpp
	$(CXX) -std=c++11 testShepard.cpp $(CXX_FLAGS) $(INCLUDES) -o testShepard -I/opt/local/include -L/opt/local/lib -lpng $(LIB_TARGET) -I../../ext/dlib

mod: createModVolDataEval.cpp
	$(CXX) createModVolDataEval.cpp $(CXX_FLAGS) $(INCLUDES) -o createModVolDataEval 

dodownsample: downsample.h doDownsample.cpp
	$(CXX) -std=c++11 doDownsample.cpp $(CXX_FLAGS) $(INCLUDES) -o downsample $(LIB_TARGET) -lpng -lpthread -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include -lbz2

downsample: downsample.h testDownsample.cpp
	$(CXX) -std=c++11 testDownsample.cpp $(CXX_FLAGS) $(INCLUDES) -I/usr/local/cuda-7.0/samples/common/inc -I/usr/local/cuda-7.0/include -o downsample
resample: resample.h doResample.cpp
	$(CXX) doResample.cpp $(CXX_FLAGS) $(INCLUDES) -I/usr/local/cuda-7.0/samples/common/inc -I/usr/local/cuda-7.0/include -o resample

test_vdhc: testVolDataHandlerCUDA.cpp lib
	$(CXX) testVolDataHandlerCUDA.cpp $(CXX_FLAGS) $(INCLUDES) -o test_vdhc -I/usr/local/cuda-6.0/samples/common/inc -I/usr/local/cuda-6.0/include -L. -lVolData

lammps: testLoadVolLammps.cpp loadVolLammpsDumpText.h
	$(CXX) testLoadVolLammps.cpp $(CXX_FLAGS) $(INCLUDES) -o testLoadVolLammps
clean:
	rm -rf *~ $(LIB_TARGET) *.o *.dSYM test_vdhc convert_traw3D

rebuild: clean all
