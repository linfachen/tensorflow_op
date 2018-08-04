TF_INCLUDE = /home/spinors/.local/lib/python3.6/site-packages/tensorflow/include
LIB_PATH = /home/spinors/.local/lib/python3.6/site-packages/tensorflow
LIB = -ltensorflow_framework

Mysqure:Mysqure.cc
	g++ -std=c++11 -shared Mysqure.cc -o Mysqure.so -fPIC -I$(TF_INCLUDE)  -L$(LIB_PATH) $(LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


