UNAME_S := $(shell uname -s)

PYTHON?=python3.11

INCLUDE=

DARWIN_EXTRA_FLAGS=
ifeq ($(UNAME_S),Linux)	
	CXX?=g++
	HOME=/home/freysn
	INCLUDE +=-I/home/freysn/dev/flotis
endif

ifeq ($(UNAME_S),Darwin)
	CXX=clang++-mp-14
	INCLUDE +=-I/Users/freysn/dev/flotis_fysn
	INCLUDE +=-I/opt/local/include
	DARWIN_EXTRA_FLAGS += -undefined dynamic_lookup -mlinker-version=450
endif

all:

	${CXX} -lc++ -g -std=c++17 -shared -fPIC ${DARWIN_EXTRA_FLAGS} ${INCLUDE} `${PYTHON} -m pybind11 --includes` timesel.cpp  -o helper_timesel`${PYTHON}-config --extension-suffix`

test:	timesel.h timesel.cpp
	${CXX} -g -std=c++17 -D_TEST_${INCLUDE} `python3.11 -m pybind11 --includes` timesel.cpp  -o test_timesel


# $(CXX) $(CXX_FLAGS) -shared -fPIC ${DARWIN_EXTRA_FLAGS} `${PYTHON} -m pybind11 --includes` supertiles_pybind11.cpp -o ldg`${PYTHON}-config --extension-suffix` -I. ${INCLUDES} -I/opt/local/include -I$(MHOME)/dev/ext/pybind11/include $(LIB)

#-I.. -I../build/_deps/helper-src/


#/opt/local/bin/clang++-mp-14 -lc++ -std=c++17 -Wall -fopenmp -O2 -shared -fPIC  -undefined dynamic_lookup -mlinker-version=450 `python3.11 -m pybind11 --includes`tss_pybind11.cpp -o flotis`python3.11-config --extension-suffix` -I. -I. -I/opt/local/include -I/opt/local/include -I/home/freysn/dev/ext/pybind11/include -lbz2 -L/opt/local/lib -lpng -lcairo -I.. -I../build/_deps/helper-src/
