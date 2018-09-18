TF_INCLUDE = /home/spinors/.local/lib/python3.6/site-packages/tensorflow/include
LIB_PATH = /home/spinors/.local/lib/python3.6/site-packages/tensorflow
LIB = -ltensorflow_framework

Mysqure:Mysqure.o mul.o
	g++ -shared Mysqure.o mul.o -o Mysqure.so -L$(LIB_PATH) $(LIB) 

Mysqure.o:Mysqure.cc
	g++ -std=c++11 -fPIC Mysqure.cc -c -o Mysqure.o  -I$(TF_INCLUDE) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

mul.o:mul.cu
	nvcc mul.cu --compiler-options "-fPIC" -c -o mul.o	


