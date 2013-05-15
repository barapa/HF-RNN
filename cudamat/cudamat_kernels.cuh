#ifndef CUDAMAT_KERNEL_H_
#define CUDAMAT_KERNEL_H_

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      100

#define COPY_BLOCK_SIZE                     16
#
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#define PI 3.1415926535897932f

__global__ void cudamat_kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);
__global__ void cudamat_kRandomUniform(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements);
__global__ void cudamat_kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements);

__global__ void cudamat_kGetRowSlice(float* source, float* target, int start, int end, int width, int height);
__global__ void cudamat_kTranspose(float *odata, float *idata, int width, int height);
__global__ void cudamat_kSetRowSlice(float* source, float* target, int start, int end, int width, int height);

__global__ void cudamat_kLessThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void cudamat_kLessThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void cudamat_kGreaterThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void cudamat_kGreaterThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void cudamat_kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void cudamat_kRowArgmax(float* mat,  float* target_max, float* target_argmax, unsigned int width, unsigned int height);
__global__ void cudamat_kSign(float* mat, float* target, unsigned int len);
__global__ void cudamat_kApplySigmoid(float* mat, float* target, unsigned int len);
__global__ void cudamat_kApplyTanh(float* mat, float* target, unsigned int len);
__global__ void cudamat_kApplyAbs(float* mat, float* target, unsigned int len);
__global__ void cudamat_kApplyLog1PlusExp(float* mat, float* target, unsigned int len);
__global__ void cudamat_kLog(float* mat, float* target, unsigned int len);
__global__ void cudamat_kExp(float* mat, float* target, unsigned int len);
__global__ void cudamat_kSqrt(float* mat, float* target, unsigned int len);
__global__ void cudamat_kPow(float* mat, float pow, float* target, unsigned int len);
__global__ void cudamat_kPowMatrix(float* mat, float* pow, float* target, unsigned int len);
__global__ void cudamat_kReciprocal(float* mat, float* target, unsigned int len);
__global__ void cudamat_kAddColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void cudamat_kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void cudamat_kAddColMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height);
__global__ void cudamat_kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void cudamat_kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void cudamat_kAdd(float* a, float* b, float* dest, unsigned int numEls);
__global__ void cudamat_kSubtract(float* a, float* b, float* dest, unsigned int numEls);
__global__ void cudamat_kMult(float* a, float* b, float* dest, unsigned int numEls);
__global__ void cudamat_kDivide(float* a, float* b, float* dest, unsigned int numEls);
__global__ void cudamat_kMultScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void cudamat_kAssignScalar(float* dest, float alpha, unsigned int len);
__global__ void cudamat_kDivideScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void cudamat_kAddScalar(float* a, float alpha, float* dest, unsigned int numEls);
__global__ void cudamat_kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows);
__global__ void cudamat_kClfVsProduct(int nComponents, int vectorLength, int nothingIndex_scalars, float* inVectors, float* outVectors, float* globalScalars, float* inIndices, int nBlocks);
__global__ void cudamat_kClfPcOuterProduct(int maxNIndexPairs, float* GindexPairs, float* nIndexPairss, float* A, float* B, float* ret, int nCols, int nBlocks);
__global__ void cudamat_kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows);
#endif
