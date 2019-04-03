CXX_OPT_DBG_FLAGS = -O3
#CXX_OPT_DBG_FLAGS = -g

CXX = g++ $(CXX_OPT_DBG_FLAGS)


CXX_FLAGS = -Wall -Wno-deprecated -Wno-sign-compare -fPIC
#-fopenmp

INCLUDES := -Ilz4

LIB_TARGET = libLosslessCompression.a

CCFILES = lz4.cpp
#CFILES = lz4/lz4.c

#OBJS :=  $(patsubst %.cpp, %.o,$(notdir $(CCFILES)))
OBJS :=  $(patsubst %.cpp, %.o,$(CCFILES))
OBJS +=  $(patsubst %.c, %.o,$(CFILES))

#$(CXX) -shared $(CXX_FLAGS) -o $(LIB_TARGET) $(LIB) $(OBJS)

OMP_FLAG = -fopenmp
#OMP_FLAG =

all: $(LIB_TARGET)

$(LIB_TARGET): $(OBJS)
	ar rcs $(LIB_TARGET) $(LIB) $(OBJS)

%.o : %.cpp
	$(CXX) -fPIC $(CXX_FLAGS) $(INCLUDES) -I$(CUDA_INC) -I$(CUDA_SDK_INC) -o $@ -c $<

test: $(LIB_TARGET) testLosslessCompression.cpp compression.h
	$(CXX) testLosslessCompression.cpp $(CXX_FLAGS) $(INCLUDES) $(LIB_TARGET) bcl/libbcl.a -o test $(OMP_FLAG) -std=c++11

test_omp: $(LIB_TARGET) testLosslessCompression_omp.cpp compression_omp.h
	$(CXX) testLosslessCompression_omp.cpp $(CXX_FLAGS) $(INCLUDES) $(LIB_TARGET) bcl/libbcl.a -o test_omp $(OMP_FLAG) -std=c++11

clean:
	rm -rf *~ $(LIB_TARGET) *.o *.dSYM lz4/*.o *.a

rebuild: clean all
