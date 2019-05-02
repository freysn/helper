OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

#all: conv_dmp2raw
#all: testMarchingSquares


INCLUDES=-I/opt/local/include -I../volData -I. -I..
LIB = -L/opt/local/lib -lpng -lbz2 -lpthread

CXX=g++
ifneq ($(DARWIN),)
CXX=/opt/local/bin/clang-mp-7.0 -lc++
endif

#CXX=/opt/ext/gcc5_svn_bin/bin/g++

CXX += -DNO_FS3D


all: som

.PHONY: img2raw conv_raw testCairoDraw conv_dmp2raw

testStatistics:
	${CXX} -g -std=c++14 -DM_VEC ${INCLUDES} ${LIB} test_statistics.cpp -o test_statistics

testSelectDifferentEntries:
	g++ -g -std=c++14 ${INCLUDES} ${LIB} test_selectDifferentEntries.cpp -o test_selectDifferentEntries

testMarchingSquares:
	g++ -g -std=c++14 ${INCLUDES} ${LIB} test_marchingSquares.cpp -o test_marchingSquares -lcairo -I../../../Dropbox/dev/play/marchingSquares

testMathFuncs:
	g++ -g -std=c++14 ${INCLUDES} ${LIB} test_mathFuncs.cpp -o test_mathFuncs

testCairoDraw:
	g++ -std=c++17 ${INCLUDES} test_CairoDraw.cpp -o test_CairoDraw -L/opt/local/lib -lcairo
img2raw:
	g++ -std=c++11 ${INCLUDES} ${LIB} app_helper_img2raw.cpp -o img2raw

conv_raw:
	${CXX} -std=c++11 ${INCLUDES} ${LIB} app_helper_conv_raw.cpp -o conv_raw

conv_dmp2raw:
	${CXX} -std=c++11 ${INCLUDES} ${LIB} app_helper_conv_DMP2raw.cpp -o conv_dmp2raw

testThreadHandler:
	${CXX} -std=c++11 test_ThreadHandler.cpp -o test_ThreadHandler -lpthread

som:
	${CXX} -std=c++14 test_SOM.cpp -o test_SOM -I.. -g
