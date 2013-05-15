#define FULL
/*
 * testconv.cu
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifdef FULL
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>

//#include "testconv_extras.cuh"
#include "conv.cuh"
//#include "conv2.cuh"
#include "conv_util.cuh"
#include "conv3.cuh"
#include "convCPU.h"
#include "gpu_locking.h"

static uint timer;

void init_tests(int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
    NVMatrix::initDeviceProps();
    NVMatrix::initRandom(7);
    //cutilCheckError(cutCreateTimer( &timer));
}

void test_convolve(int imgSize, int filterSize, bool color) {
    printf("===============================\n");
    printf("test_convolve\n");
    printf("===============================\n");

    ORDER order = IMAGE_GROUP_FILTER;
    int numFiltersPerGroup = 64, numImgsPerGroup = 128, numGroups = 4;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
//    assert(numFiltersPerGroup % 8 == 0);
    printf("Groups: %d\n", numGroups);
    printf("Images: %d, filters: %d\n", numImgsPerGroup, numFiltersPerGroup);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Output grid: %dx%d\n", numOutputsX, numOutputsX);
    printf("Color: %s\n", color ? "yes" : "no");

    int colorMult = color ? 3 : 1;
    Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
    Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
    Matrix targets(order == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
                   order == GROUP_FILTER_IMAGE ? numImgsPerGroup * numOutputs   : numFiltersPerGroup * numOutputs);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    //    cutilCheckError(cutResetTimer(timer));
    //    cutilCheckError(cutStartTimer(timer));
    if(color) {
        if (order == GROUP_FILTER_IMAGE) {
            convColorCPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        } else {
            convColorCPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        }
    } else {
        if (order == GROUP_FILTER_IMAGE) {
            convCPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        } else {
            convCPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        }
    }
    //cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);

    printf("CPU is done.\n");
    //printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    //    cutilCheckError(cutResetTimer(timer));
    //    cutilCheckError(cutStartTimer(timer));

    convolve(&nvImages, &nvFilters, &nvTargets, numGroups, color, order);

    cudaThreadSynchronize();
    //cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    //printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));
    printf("GPU is done.\n");

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

#endif //FULL




#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include "cudamat_kernels.cuh"
#include "cudamat.cuh"

extern "C" {

  // just do it, man. It'll be perfectly fine. Don't give up. 
  extern int conv(cudamat* Images, cudamat* Filters, cudamat* Targets, int numGroups, int color, int iorder){    
    ORDER order = iorder==0 ? GROUP_FILTER_IMAGE : IMAGE_GROUP_FILTER;

    //printf("Images ->size[1]=%d, Images ->size[0]=%d\n", Images ->size[1], Images ->size[0]);
    //printf("Filters ->size[1]=%d, Filters ->size[0]=%d\n", Filters ->size[1], Filters ->size[0]);

    NVMatrix nvImages (Images ->data_device, Images ->size[1], Images ->size[0], false);
    NVMatrix nvFilters(Filters->data_device, Filters->size[1], Filters->size[0], false);// is_trans=false
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);

    //printf("nvImages.getNumCols()=%d, nvImages.getNumRows()=%d\n", nvImages.getNumCols(), nvImages.getNumRows());
    //printf("nvFilters.getNumCols()=%d, nvFilters.getNumRows()=%d\n", nvFilters.getNumCols(), nvFilters.getNumRows());


    convolve(&nvImages, &nvFilters, &nvTargets, numGroups, color, order);

    cudaThreadSynchronize();


    return 0;
  }
  extern int conv2(cudamat* Images, cudamat* Filters, cudamat* Targets, int filterSize, int numGroups, int color, int iorder){    
    ORDER order = iorder==0 ? GROUP_FILTER_IMAGE : IMAGE_GROUP_FILTER;


    NVMatrix nvImages (Images ->data_device, Images ->size[1], Images ->size[0], false);
    NVMatrix nvFilters(Filters->data_device, Filters->size[1], Filters->size[0], false);// is_trans=false
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);

    //printf("nvImages.getNumCols()=%d, nvImages.getNumRows()=%d\n", nvImages.getNumCols(), nvImages.getNumRows());
    //printf("nvFilters.getNumCols()=%d, nvFilters.getNumRows()=%d\n", nvFilters.getNumCols(), nvFilters.getNumRows());

    convolve2(&nvImages, &nvFilters, &nvTargets, filterSize, numGroups, color, order);

    cudaThreadSynchronize();


    return 0;
  }

  extern int conv3(cudamat* Images, cudamat* Filters, cudamat* Targets, int numGroups, int color, int iorder){    
    ORDER order = iorder==0 ? GROUP_FILTER_IMAGE : IMAGE_GROUP_FILTER;

    NVMatrix nvFilters(Filters->data_device, Filters->size[1], Filters->size[0], false);// is_trans=false
    NVMatrix nvImages(Images->data_device, Images->size[1], Images->size[0], false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);

    convolve3(&nvImages, &nvFilters, &nvTargets, numGroups, color, order);

    cudaThreadSynchronize();


    return 0;
  }

  extern int rot180(cudamat* Filters, cudamat* Targets, int color){
    NVMatrix nvFilters(Filters->data_device, Filters->size[1], Filters->size[0], false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    rotate180(&nvFilters, &nvTargets, color);

    cudaThreadSynchronize();

    return 0;
  }

  extern int copy_into_center(cudamat* Images, cudamat* Targets, int paddingSize, int color){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    copyInto(&nvImages, &nvTargets, paddingSize, color);

    cudaThreadSynchronize();

  }

  extern int add_into_center(cudamat* Images, cudamat* Targets, int paddingSize, int color){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    addInto(&nvImages, &nvTargets, paddingSize, color);

    cudaThreadSynchronize();

  }

  extern int copy_out_of_center(cudamat* Images, cudamat* Targets, int paddingSize, int color){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    copyOutOf(&nvImages, &nvTargets, paddingSize, color);

    cudaThreadSynchronize();

  }

  extern int add_out_of_center(cudamat* Images, cudamat* Targets, int paddingSize, int color){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    addOutOf(&nvImages, &nvTargets, paddingSize, color);

    cudaThreadSynchronize();

  }

  //
  extern int sub_sample(cudamat* Images, cudamat* Targets, int factor){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    subsample(&nvImages, &nvTargets, factor);

    cudaThreadSynchronize();

  }

  extern int super_sample(cudamat* Images, cudamat* Targets, int factor){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    supersample(&nvImages, &nvTargets, factor);

    cudaThreadSynchronize();

  }
  //

  extern int matrix_to_grid(cudamat* Images, cudamat* Targets, int squareSize){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    matrixToGrid(&nvImages, &nvTargets, squareSize, true);

    cudaThreadSynchronize();

  }

  extern int grid_to_matrix(cudamat* Images, cudamat* Targets, int squareSize){
    NVMatrix nvImages( Images->data_device,  Images->size[1],  Images->size[0],  false);
    NVMatrix nvTargets(Targets->data_device, Targets->size[1], Targets->size[0], false);
    gridToMatrix(&nvImages, &nvTargets, squareSize, true);

    cudaThreadSynchronize();

  }



  //#ifdef FULL
extern int run_test_convolve(){
    int boardNum = get_board_lock();
    if (boardNum == GPU_LOCK_NO_BOARD) {
        printf("No free GPU boards!\n");
        exit(EXIT_FAILURE);
    } else if(boardNum == GPU_LOCK_NO_SCRIPT) {
        printf("Running on default board.\n");
    } else {
        printf("Running on board %d\n", boardNum);
    }
    init_tests(boardNum);

    test_convolve(32, 9, true);
    return 0;
}
  //#endif

/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

extern int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

extern int cublas_shutdown() {
    cublasShutdown();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}


extern int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int init_random(rnd_struct* rnd_state, int seed, char* cudamatpath) {
    unsigned int * host_mults;
    host_mults = (unsigned int*)malloc(NUM_RND_STREAMS * sizeof(unsigned int));
    FILE * pFile;

    pFile = fopen (cudamatpath,"r");

    for (int i = 0; i < NUM_RND_STREAMS; i++) {
        fscanf (pFile, "%u", &host_mults[i]);
    }
    fclose (pFile);

    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned int), (void**)&rnd_state->dev_mults);
    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned long long), (void**)&rnd_state->dev_words);
    cublasSetVector(NUM_RND_STREAMS, sizeof(unsigned int), host_mults, 1, rnd_state->dev_mults, 1);
    //cudaMalloc((void **)&rnd_state->dev_mults, NUM_RND_STREAMS * sizeof(unsigned int));
    //cudaMalloc((void **)&rnd_state->dev_words, NUM_RND_STREAMS * sizeof(unsigned long long));
    //cudaMemcpy(rnd_state->dev_mults, host_mults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    cudamat_kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, seed);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

/* ------------------------------ Utility routines ------------------------------ */

extern int get_leading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

extern int get_nonleading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

extern void set_transpose(cudamat* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(cudamat* mat) {
    return mat->is_trans ? 't' : 'n';
}

extern void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

extern int allocate_device_memory(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    cublasStatus stat;

    stat = cublasAlloc(len, sizeof(mat->data_device[0]), (void**)&mat->data_device);

    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    mat->on_device = 1;
    return 0;
}

extern int copy_to_host(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    if (mat->on_device) {
            cublasGetVector(len, sizeof(mat->data_host[0]), mat->data_device, 1, mat->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}




extern int copy_to_device(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];
    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(mat->data_host[0]), mat->data_host, 1, mat->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

extern int copy_on_device(cudamat* mat1, cudamat* mat2) {
    int len = mat1->size[0]*mat1->size[1];

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasScopy(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

extern int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = source->size[0];
    int width = source->size[1];

    if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    cudamat_kGetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int set_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = target->size[0];
    int width = target->size[1];

    if ((end - start) != source->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    cudamat_kSetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int copy_transpose(cudamat* source, cudamat* target) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    cudamat_kTranspose<<< grid, threads >>>(target->data_device, source->data_device, height, width);

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int free_device_memory(cudamat* mat) {
    if (mat->owns_data && mat->on_device) {
        cublasStatus stat;

        stat = cublasFree(mat->data_device);
        mat->on_device = 0;

        if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
            return CUBLAS_ERROR;
    }

    return 0;
}

extern int reshape(cudamat* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

extern int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_col * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;

    return 0;
}

extern int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_ind * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;

    if (source->size[0] > 1) {
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
    }

    return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

///// ADDED BY IS FOR THEANO COMPATIBILITY
extern void init_from_cuda_ndarray(cudamat* mat, long gpu_pointer, int m, int n){
  mat->on_host=0;
  mat->size[0]=m;
  mat->size[1]=n;
  mat->on_device=1;
  mat->is_trans=0;
  mat->owns_data=0; 
  mat->data_device = (float*) ((void*) gpu_pointer);
}

extern void init_from_array(cudamat* mat, float* data, int m, int n) {
    mat->data_host = data;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

extern int init_empty(cudamat* mat, int m, int n) {
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */
extern int fill_with_rand(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    cudamat_kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int fill_with_randn(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    cudamat_kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
/* ------------------------------ Algebraic operations ------------------------------ */

extern int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kAddColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kAddColMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, mult, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kAddRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kMultByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kMultByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int less_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kLessThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int less_than_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kLessThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kGreaterThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int greater_than_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kGreaterThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int max_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        cudamat_kMaxColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int max_row_argmax(cudamat* mat, cudamat* target_max, cudamat* target_argmax, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target_max->on_device || !target_argmax->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target_max->size[0] != 1 || 
	    target_max->size[1] != mat->size[1] || 
	    target_max->size[1] != target_argmax->size[1] ||
	    mat->size[0] != target_argmax->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        cudamat_kRowArgmax<<<w,32>>>(mat->data_device, 
				     target_max->data_device, 
				     target_argmax->data_device, 
				     w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}



extern int sign(cudamat* mat, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kSign<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_sigmoid(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kApplySigmoid<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_tanh(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kApplyTanh<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_abs(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kApplyAbs<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_log_1_plus_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kApplyLog1PlusExp<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_log(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kLog<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kExp<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_sqrt(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kSqrt<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_pow(cudamat* mat, float pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kPow<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kPowMatrix<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int reciprocal(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kReciprocal<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int dot(cudamat* mat1, cudamat* mat2, cudamat* target, float beta, float alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

    cublasSgemm(get_transpose_char(mat1), get_transpose_char(mat2), 
                m, n, k,
                alpha, mat1->data_device, mat1->size[0],
                mat2->data_device, mat2->size[0],
                beta, target->data_device, target->size[0]);

    cudaThreadSynchronize();

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

extern float vdot(cudamat* mat1, cudamat* mat2, int* err_code) {
    int len = mat1->size[0]*mat1->size[1];
    float res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }

    res = cublasSdot(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
   have the same transposedness. */
extern int add_mult(cudamat* mat1, cudamat* mat2, float alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasSaxpy(len, alpha, mat2->data_device, 1, mat1->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

extern int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kAdd<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kSubtract<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kDivide<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Elementwise multiplication of 2 matrices */
extern int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int assign_scalar(cudamat* mat, float alpha) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    cudamat_kAssignScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kMultScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int divide_by_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kDivideScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int add_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cudamat_kAddScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern float euclid_norm(cudamat* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];

    float res =  cublasSnrm2(len, mat->data_device, 1);

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

// extern int selectRows(cudamat* source, cudamat* target, cudamat* indices){
//     const int nRetRows = indices->size[1];

//     if (nRetRows==0) return 0;

//     dim3 gridDim((nRetRows+31)/32);
//     dim3 blockDim(32);

//     cudamat_kSelectRows<<<gridDim, blockDim>>>(source->data_device, target->data_device, indices->data_device, nRetRows, source->size[0], source->size[1]);
//     cudaThreadSynchronize();

//     if (checkCUDAError())
//         return CUDA_ERROR;
//     else
//         return 0;
// }

extern int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices){
    const int nSetRows = indices->size[1];

    if (nSetRows==0)
        return 0;

    dim3 gridDim((nSetRows+31)/32);
    dim3 blockDim(32);

    cudamat_kSetSelectedRows<<<gridDim, blockDim>>>(target->data_device, source->data_device, indices->data_device, nSetRows, target->size[0], target->size[1]);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}


extern int clfVsProduct(int nComponents, int vectorLength, int nothingIndex_scalars, cudamat* inVectors, cudamat* outVectors, cudamat* globalScalars, cudamat* inIndices, int nOutputs){
  int gridX=nOutputs, gridY=1;
  while (gridX>65535) {gridY*=2; gridX = (gridX+1)/2;}
  dim3 gridDim(gridX, gridY);
  if (nOutputs>0) cudamat_kClfVsProduct<<<gridDim, max(32, vectorLength)>>>(nComponents, vectorLength, nothingIndex_scalars,
								    inVectors->data_device, outVectors->data_device, globalScalars->data_device, inIndices->data_device, nOutputs);
  cudaThreadSynchronize();
  if (checkCUDAError()) return CUDA_ERROR; else return 0;
}


extern int clfPcOuterProduct(cudamat* indexPairs, cudamat* nIndexPairss, cudamat* A, cudamat* B, cudamat* ret){
  const int nCols = ret->size[0];
  const int nRows = ret->size[1];
  int gridX=nRows, gridY=1;
  while (gridX>65535) {gridY*=2; gridX = (gridX+1)/2;}
  dim3 gridDim(gridX, gridY);
  const int Ns = indexPairs->size[0]*4;
  cudamat_kClfPcOuterProduct<<<gridDim, max(32, nCols), Ns>>>(indexPairs->size[0]/2, indexPairs->data_device, nIndexPairss->data_device, A->data_device, B->data_device, ret->data_device, nCols, nRows);
  cudaThreadSynchronize();
  if (checkCUDAError()) return CUDA_ERROR; else return 0;
}


extern int selectRows(cudamat* source, cudamat* target, cudamat* indices){
    const int nRetRows = indices->size[1];

    if (nRetRows==0) return 0;

    int gridX=(nRetRows+31)/32, gridY=1;
    while (gridX>65535) {gridY*=2; gridX = (gridX+1)/2;}
    dim3 gridDim(gridX, gridY);

    dim3 blockDim(32);

    cudamat_kSelectRows<<<gridDim, blockDim>>>(source->data_device, target->data_device, indices->data_device, nRetRows, source->size[0], source->size[1]);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}




}









