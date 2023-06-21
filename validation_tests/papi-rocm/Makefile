CC        = $(HIP_ROOT)/bin/hipcc
CXX       = $(HIP_ROOT)/bin/hipcc
CPPFLAGS += -I$(HIP_ROOT)/include
LDFLAGS  += -L$(HIP_LIB_PATH) -L$(PAPI_LIB_PATH) -lpapi -fopenmp

ALL: omp_gpu_monitor

omp_gpu_monitor: omp_gpu_monitor.o matmul.o

clean:
	rm -f *.o omp_gpu_monitor
