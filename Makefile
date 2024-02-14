CXX=nvc++ 
ifneq (,$(findstring nvc++,$(CXX)))
	CXXFLAGS=-fast -mp -std=c++20
	LDFLAGS=
	STDPARFLAGS_GPU=-mp -stdpar=gpu -gpu=ccall
	STDPARFLAGS_CPU=-mp -stdpar=multicore
all: BlackScholes_gpu BlackScholes_cpu pnl_gpu pnl_cpu
else
	echo "Unknown Compiler"
	exit
endif


BlackScholes_gpu: BlackScholes_reference.o BlackScholes_main_gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_GPU) 

BlackScholes_cpu: BlackScholes_reference.o BlackScholes_main_cpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_CPU) 

BlackScholes_main_gpu.o: BlackScholes_main.cpp BlackScholes_stdpar.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< 

BlackScholes_main_cpu.o: BlackScholes_main.cpp BlackScholes_stdpar.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -c -o $@ $< 

pnl_gpu.o: pnl.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< 

pnl_gpu: pnl_gpu.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -o $@ $< 

pnl_cpu.o: pnl.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -c -o $@ $< 

pnl_cpu: pnl_cpu.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -o $@ $< 

BlackScholes_reference.o: BlackScholes_reference.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

.PHONY: clean
clean:
	-rm -f core *.o BlackScholes_gpu BlackScholes_cpu pnl_cpu pnl_gpu
