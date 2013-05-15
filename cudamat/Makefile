# trying to copy parts of common-gcc-cuda-2.3.mk in order to build our conv2 thing.
COMMONFLAGS := -DRND_MULTIPLIERS_FILE=\"rnd_multipliers_32bit.txt\"
ATLAS_LIB:=/ais/gobi1/pkgs/pylab-03/x86_64_GenuineIntel_10.04/ATLAS/lib/ # /usr/lib/atlas
INCLUDES := 
#LIB := -L/pkgs_local/cuda-2.3/lib64/ -L$(ATLAS_LIB) -latlas -lcblas  
LIB := -L$(ATLAS_LIB) -latlas -lcblas  
#-L/pkgs_local/cuda-sdk-2.1/lib/
ROOTDIR    ?= $(CUDA_SDK_PATH)
ROOTBINDIR ?= ./bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= ./obj
LIBDIR     := $(ROOTDIR)/C/lib
COMMONDIR  := $(ROOTDIR)/C/common
LIB       += -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) 
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc
ifeq ($(USEDRVAPI),1)
  LIB := -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
else
  LIB := -lcudart ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB}
endif
LIB += -lcublas -lcutil

COMMONFLAGS := -DUNIX

CC := g++ -fpic
NVCC := nvcc --ptxas-options=-v --compiler-options '-fPIC' $(LIB) $(INCLUDES) $(COMMONFLAGS)
OBJS := matrix.o convCPU.o gpu_locking.o nvmatrix.o nvmatrix_kernel.o conv.o conv2.o conv3.o conv_util.o	
CUDAMAT_OBJS := cudamat_kernels.o cudamat.o


leftovers:
	$(CC) matrix.cpp -c -o matrix.o
objs:
	$(CC) convCPU.cpp -c -o convCPU.o
	$(CC) gpu_locking.cpp -c -o gpu_locking.o
	$(NVCC) nvmatrix.cu  -c -o nvmatrix.o
	$(NVCC) nvmatrix_kernel.cu -c -o nvmatrix_kernel.o
	$(NVCC) conv.cu -c -o conv.o
	$(NVCC) conv2.cu -c -o conv2.o
	$(NVCC) conv3.cu -c -o conv3.o
	$(NVCC) conv_util.cu -c -o conv_util.o	

cudamat_objs:
	$(NVCC) cudamat_kernels.cu  -c -o cudamat_kernels.o
	$(NVCC) cudamat.cu -c -o cudamat.o


cudamat:
	#nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu cudamat_kernels.cu -L/pkgs_local/cuda-3.2/lib64/ -lcublas		
        #nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu cudamat_kernels.cu -L/pkgs_local/cuda/lib/ -lcublas
	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared  $(OBJS) $(CUDAMAT_OBJS)   -lcublas 

	#nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudalearn.so --shared learn.cu learn_kernels.cu -L/pkgs_local/cuda/lib/ -lcublas
	#nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudalearn.so --shared learn.cu learn_kernels.cu -lcublas


clean:
	rm *.linkinfo *.pyc *.so
