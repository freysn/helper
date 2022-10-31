#F=play_composit
#F=play_colormaps
# F=play_mvec

# all:
# 	g++ -O3 -openmp -std=c++11 ${F}.cpp -I/opt/local/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/include -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include  -L/opt/local/lib -lpng -o ${F}
# 	./${F}

all:
	/opt/local/bin/clang-mp-4.0 -std=c++14  -lc++ -L/opt/local/lib cm_convert.cpp -o cm_convert -I.. -I.
