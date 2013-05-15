#include "cudamat_kernels.cuh"
#include "float.h"

/* ------------------------- Random number generation ------------------------- */

__global__ void cudamat_kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The initial x is the seed and the initial carry is 1
    unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;
    const unsigned int rndMult = rndMults[idx];
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar.
     */
    for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    }
    rndWords[idx] = rndWord;
}

__global__ void cudamat_kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

__global__ void cudamat_kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

/* ------------------------- Data copying ------------------------- */

/*
Copy row slice from source to target. There is a block for every 32x32 chunk being copied.
*/
__global__ void cudamat_kGetRowSlice(float* source, float* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int target_height = end - start;

    if (row < end) {
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * target_height + row - start] = source[cur_col * height + row];
    }
}

__global__ void cudamat_kSetRowSlice(float* source, float* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int source_height = end - start;

    if (row < end) {
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * height + row] = source[cur_col * source_height + row - start];
            //source[cur_col * height + row - start] = target[cur_col * target_height + row];
    }
}

__global__ void cudamat_kTranspose(float *odata, float *idata, int width, int height) {
    __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------- Mathematical operations ------------------------- */

__global__ void cudamat_kLessThan(float* mat1, float* mat2, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat1[i] < mat2[i];
    }
}

__global__ void cudamat_kLessThanScalar(float* mat, float val, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] < val;
    }
}

__global__ void cudamat_kGreaterThan(float* mat1, float* mat2, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat1[i] > mat2[i];
    }
}

__global__ void cudamat_kGreaterThanScalar(float* mat, float val, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] > val;
    }
}

__global__ void cudamat_kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    float cur_max = -FLT_MAX;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
}


__global__ void cudamat_kRowArgmax(float* mat, 
				   float* target_max, 
				   float* target_arg_max,
				   unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    __shared__ float max_inds[32];
    float cur_max = -FLT_MAX;
    int ind_max = 0;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val > cur_max){
            cur_max = val;
	    ind_max = i;
	}
    }

    max_vals[threadIdx.x] = cur_max;
    max_inds[threadIdx.x] = ind_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
	  if (max_vals[i] > cur_max){
                cur_max = max_vals[i];
		ind_max = max_inds[i];
	  }

        target_max[blockIdx.x] = cur_max;
	target_arg_max[blockIdx.x * height + ind_max] = 1; // that's the way to go. Yes. Now we have the argmax.
    }
}


__global__ void cudamat_kSign(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] ? copysignf(1., mat[i]) : 0.;
    }
}

__global__ void cudamat_kApplySigmoid(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = 1 / (1 + __expf(-mat[i]));
    }
}


__global__ void cudamat_kApplyTanh(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    float mat_i, exp2x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        exp2x = __expf(2 * mat_i);
        target[i] = 1 - 2 / (exp2x + 1);
    }
}

__global__ void cudamat_kApplyAbs(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] * ((mat[i] > 0) - (mat[i] < 0));
    }
}

__global__ void cudamat_kApplyLog1PlusExp(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    float mat_i;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        if (mat_i > 0)
            target[i] = (__logf(1 + __expf(-mat_i)) + mat_i);
        else
            target[i] = __logf(1 + __expf(mat_i));
    }
}

__global__ void cudamat_kLog(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = __logf(mat[i]);
    }
}

__global__ void cudamat_kExp(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = __expf(mat[i]);
    }
}

__global__ void cudamat_kSqrt(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = sqrt(mat[i]);
    }
}

__global__ void cudamat_kPow(float* mat, float pow, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = powf(mat[i], pow);
    }
}

__global__ void cudamat_kPowMatrix(float* mat, float* pow, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = powf(mat[i], pow[i]);
    }
}

__global__ void cudamat_kReciprocal(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads)
        target[i] = 1. / mat[i];
}

__global__ void cudamat_kAddColVector(float* mat, float* vec, float* tgtMat, unsigned int width,
                              unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i % height];
    }
}

__global__ void cudamat_kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i / height];
    }
}

__global__ void cudamat_kAddColMult(float* mat, float* vec, float* tgtMat, float mult,
                            unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + mult * vec[i % height];
    }
}

__global__ void cudamat_kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i % height];
    }
}

__global__ void cudamat_kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i / height];
    }
}

__global__ void cudamat_kAdd(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + b[i];
    }
}

__global__ void cudamat_kSubtract(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] - b[i];
    }
}

__global__ void cudamat_kDivide(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] / b[i];
    }
}

__global__ void cudamat_kMult(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] * b[i];
    }
}

__global__ void cudamat_kMultScalar(float* mat, float alpha, float* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha * mat[i];
    }
}

__global__ void cudamat_kAssignScalar(float* dest, float alpha, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha;
    }
}

__global__ void cudamat_kDivideScalar(float* mat, float alpha, float* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = mat[i] / alpha;
    }
}

__global__ void cudamat_kAddScalar(float* a, float alpha, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + alpha;
    }
}

// __global__ void cudamat_kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows){
//     __shared__ int sourceRowIndices[32];
//     const int startTargetRowI = blockIdx.x * 32;
//     const int tid = threadIdx.x;
//     const int localNRowIs = min(32, nRowIs-startTargetRowI);

//     // cooperatively load 32 row indices
//     if (tid < localNRowIs){
//         sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
//         if (sourceRowIndices[tid]<0)
//             sourceRowIndices[tid] += nSourceRows;
//         if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
//             sourceRowIndices[tid] = -1;
//     }
//     __syncthreads();

//     // copy 32 rows
//     for (int i=0; i<localNRowIs; i++){
//         const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
//         for (int colI=tid; colI<nCols; colI+=32)
//             target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
//     }
// }

__global__ void cudamat_kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows){
    __shared__ int targetRowIndices[32];
    const int startSourceRowI = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int localNRowIs = min(32, nRowIs-startSourceRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        targetRowIndices[tid] = int(indices[startSourceRowI + tid]);
        if (targetRowIndices[tid]<0)
            targetRowIndices[tid] += nTargetRows;
        if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nTargetRows)
            targetRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    for (int i=0; i<localNRowIs; i++){
        const int sourceRowI = startSourceRowI + i, targetRowI = targetRowIndices[i];
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}


__global__ void cudamat_kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows){
    __shared__ int sourceRowIndices[32];
    const int tid = threadIdx.x, bid = blockIdx.x*gridDim.y+blockIdx.y;
    const int startTargetRowI = bid * 32;
    if (startTargetRowI>=nRowIs) return; // unneeded block
    const int localNRowIs = min(32, nRowIs-startTargetRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
        if (sourceRowIndices[tid]<0)
            sourceRowIndices[tid] += nSourceRows;
        if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
            sourceRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    for (int i=0; i<localNRowIs; i++){
        const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}

__global__ void cudamat_kClfPcOuterProduct(int maxNIndexPairs, float* GindexPairs, float* nIndexPairss, float* A, float* B, float* ret, int nCols, int nBlocks){
  /*
    a block per row. a thread per ret cell (except: at least 32 threads).
    dynamically allocated shared mem: two words per index pair.
  */
  const int tid = threadIdx.x, bid = blockIdx.x*gridDim.y+blockIdx.y;
  if (bid>=nBlocks) return;

  const int nIndexPairs = nIndexPairss[bid];
  extern __shared__ int indexPairs[];

  // cooperatively load index pairs
  GindexPairs += bid * maxNIndexPairs*2;
  const int nValuesToCopy = nIndexPairs * 2;
  if (tid<32) for (int i=tid; i<nValuesToCopy; i+=32) indexPairs[i] = GindexPairs[i];
  __syncthreads();

  // from now on, everything is thread local
  if (tid<nCols){
    float cum=0;
    for (int iPairI=0; iPairI<nIndexPairs; iPairI++){
      const int aI = indexPairs[iPairI*2], bI = indexPairs[iPairI*2+1];
      cum += A[nCols*aI + tid] * B[nCols*bI + tid];
    }
    ret[nCols*bid + tid] = cum;
  }
}

__global__ void cudamat_kClfVsProduct(int nComponents, int vectorLength, int nothingIndex_scalars, float* inVectors, float* outVectors, float* globalScalars, float* inIndices, int nBlocks){
  __shared__ int vectorIndices[32];
  __shared__ float localScalars[32];
  
  const int tid = threadIdx.x, bid = blockIdx.x*gridDim.y+blockIdx.y;
  if (bid>=nBlocks) return;
  inIndices += bid * 2 * nComponents;
  
  float cum=0;
  for (int componentI=0; componentI<nComponents; componentI++){
    if ((componentI & 31) == 0){ // download the next 32 scalars & indices
      if (tid<32 && componentI+tid<nComponents){
	vectorIndices[tid] = inIndices[componentI+tid];
	const int scalarIndex = inIndices[componentI+tid + nComponents];
	localScalars[tid] = (scalarIndex==nothingIndex_scalars) ? 0 : globalScalars[scalarIndex];
      }
    }
    __syncthreads(); // this should of course be in the conditional that just closed, but putting it there seems to make it not work.
    if ((tid < vectorLength) && (localScalars[componentI & 31] != 0))
      cum += localScalars[componentI & 31] * inVectors[vectorIndices[componentI & 31] * vectorLength + tid];
  }
  if (tid < vectorLength)
    outVectors[bid * vectorLength + tid] = cum;
}
    

