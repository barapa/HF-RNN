/*
 * conv3.cuh
 * CUDA 2.1-2.3 convolution routines optimized for GT200 architecture.
 *
 * Nov 15, 2009
 * Alex Krizhevsky (akrizhevsky@gmail.com)
 *
 * The routines in this file are useful for sampling the visible units of a convolutional RBM
 * given the hiddens. See conv3CPU for the exact operation that they perform.
 */

#ifndef CONV3_CUH_
#define CONV3_CUH_

#include <cutil_inline.h>
#include <assert.h>
#include <matrix.h>
#include <nvmatrix.cuh>
#include "conv_common.cuh"

void convolve3(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int numGroups, bool color, ORDER imgOrder = GROUP_FILTER_IMAGE);

/*
 * This function uses block size 16x16.
 * Works for filters up to 37x37.
 */
template<int filterSize, bool checkBounds, int stride, bool gfi>
__global__ void conv3_bw_fit_16x16(float* imgs, float* filters, float* targets,
                                   const int imgSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 15;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[filterSize][filterSize];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int outputPartY = outputPart / DIVUP(numOutputsX, 16);
    const int outputPartX = outputPart % DIVUP(numOutputsX, 16);
    const int tidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 16x16 "plate" of threads

    const int imgPixels = MUL24(imgSize, imgSize);

//    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    const int filterPixels = filterSize * filterSize;

    if (gfi) {
        imgs += MUL24(MUL24(groupIdx, numImgsPerGroup/stride), numFiltersPerGroup) * imgPixels
              + MUL24(imgIdxInGroup/stride, imgPixels)
              + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
    } else {
        imgs += MUL24(MUL24(imgIdxInGroup/stride, numGroups), numFiltersPerGroup) * imgPixels
              + MUL24(groupIdx, numFiltersPerGroup) * imgPixels
              + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
    }
    targets += MUL24(groupIdx, numImgsPerGroup) * numOutputs
             + MUL24(imgIdxInGroup, numOutputs)
             + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
             + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    if (filterSize <= 16)
        filters += tidx;
    filters += MUL24(groupIdx, numFiltersPerGroup) * filterPixels * stride
             + MUL24(imgIdxInGroup % stride, filterPixels);
    const float* lastFilter = filters + filterPixels * stride * numFiltersPerGroup; // bad pointer
    float prod = 0;
    const bool compute = !checkBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY*16;
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        __syncthreads();

        /*
         * It might seem strange to have all these ifs explicitly in the loops rather than
         * just looping from x = threadIdx.x to min(shImgSizeX, cmpX), but this makes the loop bounds
         * compile-time constants, which allows the compiler to unroll the inner loop.
         */
        // Load image piece into shmem
        if (checkBounds) {
            int y;
            for (y = 0; y < shImgSizeY - 16; y += 16) {
                const int loadY = threadIdx.y + y;
                if (loadY < cmpY) {
                    int x;
                    for (x = 0; x < shImgSizeX - 16; x += 16) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < cmpX) {
                            shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                        }
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX && loadX < cmpX) {
                        shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY && loadY < cmpY) {
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    if (loadX < cmpX) {
                        shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX && loadX < cmpX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        } else { // turns out this is faster than computing indices using division/mod
            int y;
            for (y = 0; y < shImgSizeY - 16; y += 16) {
                const int loadY = threadIdx.y + y;
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY) {
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        }

        // Load filter into shmem
        if (filterSize <= 16) {
            if (tidx < filterPixels)
                shFilter[0][tidx] = filters[0];
        } else {
            #pragma unroll
            for (int y = 0; y < filterSize; y += 16) {
                const int loadY = threadIdx.y + y;
                if (loadY < filterSize) {
                    for (int x = 0; x < filterSize; x += 16) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < filterSize) {
                            shFilter[loadY][loadX] = filters[MUL24(loadY, filterSize) + loadX];
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (compute) {
            const float* myShFilter = &shFilter[filterSize - 1][filterSize - 1];
            const float* myShImg = &shImg[threadIdx.y][threadIdx.x];

            if(filterSize < 16) {
                #pragma unroll // commented to speed up compiling
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];

                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 15;
                }
            } else {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];

                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 15;
                }
            }
        }
        if (gfi) {
            imgs += MUL24(numImgsPerGroup/stride, imgPixels);
        } else {
            imgs += imgPixels;
        }
        filters += filterPixels * stride;
    } while (filters != lastFilter);

    if (compute) {
        targets[0] = prod;
    }
}


/*
 * This function uses block size 16x16.
 * Use for filters > 37x37.
 */
template<bool checkOutputBounds, bool checkFilterBounds, int stride, bool gfi>
__global__ void conv3_bw_nofit_16x16(float* imgs, float* filters, float* targets,
        const int imgSize, const int filterSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = 16 * 2 - 1, shImgSizeY = 16 * 2 - 1;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][16];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int outputPartY = outputPart / DIVUP(numOutputsX, 16);
    const int outputPartX = outputPart % DIVUP(numOutputsX, 16);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image
    const int filterPixels = MUL24(filterSize, filterSize);

    if (gfi) {
        imgs += MUL24(MUL24(groupIdx, numImgsPerGroup/stride), numFiltersPerGroup) * imgPixels
              + MUL24(imgIdxInGroup/stride, imgPixels)
              + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
    } else {
        imgs += MUL24(MUL24(imgIdxInGroup/stride, numGroups), numFiltersPerGroup) * imgPixels
              + MUL24(groupIdx, numFiltersPerGroup) * imgPixels
              + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
    }
    targets += MUL24(groupIdx, numImgsPerGroup) * numOutputs
             + MUL24(imgIdxInGroup, numOutputs)
             + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
             + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    filters += MUL24(groupIdx, numFiltersPerGroup) * filterPixels * stride
             + MUL24(imgIdxInGroup % stride, filterPixels);

    const float* lastFilter = filters + MUL24(MUL24(filterPixels, stride), numFiltersPerGroup); // bad pointer, hope nothing rolls over...
    float prod = 0;
    const bool compute = !checkOutputBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY * 16;

    float* shFilterLoad = &shFilter[15 - threadIdx.y][15 - threadIdx.x];
    float* shImgLoad = &shImg[threadIdx.y][threadIdx.x];
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        for (int fY = 0; fY < filterSize; fY += 16) {
            for (int fX = 0; fX < filterSize; fX += 16) {
                __syncthreads();

                // Load image piece into shmem
                // this must exist cause f > 37 ==> i > 37

                if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY < cmpY)
                    shImgLoad[0] = imgs[MUL24(threadIdx.y + fY, imgSize) + threadIdx.x + fX];
                if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY < cmpY)
                    shImgLoad[15] = imgs[MUL24(threadIdx.y + fY, imgSize) + threadIdx.x + fX + 15];
                if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY + 15 < cmpY)
                    shImgLoad[15 * shImgSizeX] = imgs[MUL24(threadIdx.y + fY + 15, imgSize) + threadIdx.x + fX];
                if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY + 15 < cmpY)
                    shImgLoad[15 * shImgSizeX + 15] = imgs[MUL24(threadIdx.y + fY + 15, imgSize) + threadIdx.x + fX + 15];

                // Load filter piece into shmem

                const int rotFx = threadIdx.x + filterSize - fX - 16, rotFy = threadIdx.y + filterSize - fY - 16;
                if (checkFilterBounds)
                    shFilterLoad[0] = 0;
                if (!checkFilterBounds || (rotFx >= 0 && rotFy >= 0))
                    shFilterLoad[0] = filters[MUL24(filterSize, rotFy) + rotFx];

                __syncthreads();

                if (compute) {
                    const float* myShFilter = &shFilter[0][0];
                    const float* myShImg = &shImg[threadIdx.y][threadIdx.x];

                    // TODO: uncomment this in final version!
                    #pragma unroll // commented to speed up compiling
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 16; j++) {
                            prod += myShFilter[0] * myShImg[0];

                            myShFilter++;
                            myShImg++;
                        }
                        myShImg += 15;
                    }
                }
            }
        }

        if (gfi) {
            imgs += MUL24(numImgsPerGroup/stride, imgPixels);
        } else {
            imgs += imgPixels;
        }
        filters += filterPixels * stride;
    } while (filters != lastFilter);

    if (compute) {
        targets[0] = prod;
    }
}


/*
 * Like above but uses block size 8x8x4, each block doing 4 images.
 * Works for filters up to 18x18 or so.
 *
 * Surprisingly slower than above version.
 * NOTE: not used anywhere.
 */
template<int filterSize, bool checkBounds, int stride>
__global__ void conv3_bw_fit_8x8x4(float* imgs, float* filters, float* targets,
                                   const int imgSize, const int numFiltersPerGroup, const int numImgsPerGroup, const int numGroups) {
    const int shImgSizeX = filterSize + 7, shImgSizeY = filterSize + 7;
    __shared__ float shImg[4][shImgSizeY][shImgSizeX];
    __shared__ float shFilter[4][filterSize][filterSize];

//    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = (4*blockIdx.x + threadIdx.z) % numImgsPerGroup;
    const int groupIdx = (4*blockIdx.x + threadIdx.z) / numImgsPerGroup;
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int outputPartY = outputPart / DIVUP(numOutputsX, 8);
    const int outputPartX = outputPart % DIVUP(numOutputsX, 8);
    const int tidx = threadIdx.y * 8 + threadIdx.x; // thread's index within the 8x8 "plate" of threads

    const int imgPixels = MUL24(imgSize, imgSize);

//    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    const int filterPixels = filterSize * filterSize;

    imgs += MUL24(MUL24(groupIdx, numImgsPerGroup/stride), numFiltersPerGroup) * imgPixels
          + MUL24(imgIdxInGroup/stride, imgPixels)
          + MUL24(outputPartY, imgSize) * 8 + outputPartX * 8; // hid acts for conv rbm
    targets += MUL24(groupIdx, numImgsPerGroup) * numOutputs
             + MUL24(imgIdxInGroup, numOutputs)
             + MUL24(outputPartY, numOutputsX) * 8 + outputPartX * 8
             + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    if (filterSize <= 8)
        filters += tidx;
    filters += MUL24(groupIdx, numFiltersPerGroup) * filterPixels * stride
             + MUL24(imgIdxInGroup % stride, filterPixels);
    const float* lastFilter = filters + filterPixels * stride * numFiltersPerGroup; // bad pointer
    float prod = 0;
    const bool compute = !checkBounds || (outputPartX * 8 + threadIdx.x < numOutputsX && outputPartY * 8 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 8, cmpY = imgSize - outputPartY * 8;
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        __syncthreads();

        /*
         * It might seem strange to have all these ifs explicitly in the loops rather than
         * just looping from x = threadIdx.x to min(shImgSizeX, cmpX), but this makes the loop bounds
         * compile-time constants, which allows the compiler to unroll the inner loop.
         */
        // Load image piece into shmem
        if (checkBounds) {
            int y;
            for (y = 0; y < shImgSizeY - 8; y += 8) {
                const int loadY = threadIdx.y + y;
                if (loadY < cmpY) {
                    int x;
                    for (x = 0; x < shImgSizeX - 8; x += 8) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < cmpX) {
                            shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                        }
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX && loadX < cmpX) {
                        shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY && loadY < cmpY) {
                int x;
                for (x = 0; x < shImgSizeX - 8; x += 8) {
                    const int loadX = threadIdx.x + x;
                    if (loadX < cmpX) {
                        shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX && loadX < cmpX) {
                    shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        } else { // turns out this is faster than computing indices using division/mod
            int y;
            for (y = 0; y < shImgSizeY - 8; y += 8) {
                const int loadY = threadIdx.y + y;
                int x;
                for (x = 0; x < shImgSizeX - 8; x += 8) {
                    const int loadX = threadIdx.x + x;
                    shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY) {
                int x;
                for (x = 0; x < shImgSizeX - 8; x += 8) {
                    const int loadX = threadIdx.x + x;
                    shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[threadIdx.z][loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        }

        // Load filter into shmem
        if (filterSize <= 8) {
            if (tidx < filterPixels)
                shFilter[threadIdx.z][0][tidx] = filters[0];
        } else {
            #pragma unroll
            for (int y = 0; y < filterSize; y += 8) {
                const int loadY = threadIdx.y + y;
                if (loadY < filterSize) {
                    for (int x = 0; x < filterSize; x += 8) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < filterSize) {
                            shFilter[threadIdx.z][loadY][loadX] = filters[MUL24(loadY, filterSize) + loadX];
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (compute) {
            const float* myShFilter = &shFilter[threadIdx.z][filterSize - 1][filterSize - 1];
            const float* myShImg = &shImg[threadIdx.z][threadIdx.y][threadIdx.x];

            if(filterSize < 8) {
                #pragma unroll // commented to speed up compiling
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];

                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 7;
                }
            } else {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];

                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 7;
                }
            }
        }
        imgs += MUL24(numImgsPerGroup/stride, imgPixels);
        filters += filterPixels * stride;
    } while (filters != lastFilter);

    if (compute) {
        targets[0] = prod;
    }
}

/*
 * This function uses block size (y,x) = 8x16.
 * Works for filters up to 37x37.
 *
 * NOTE: not used anywhere
 */
template<int filterSize, bool checkBounds, int stride>
__global__ void conv3_bw_fit_8x16(float* imgs, float* filters, float* targets, const int imgSize, const int numFilters) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 7;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[filterSize][filterSize];

    const int caseIdx = blockIdx.x;
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int outputPartY = outputPart / DIVUP(numOutputsX, 16);
    const int outputPartX = outputPart % DIVUP(numOutputsX, 16);
    const int tidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads

    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    const int filterPixels = filterSize * filterSize;
//    const int loadX = tidx % (shImgSizeX);
//    const int loadY = tidx / (shImgSizeX);

    imgs += MUL24(numFilters, MUL24(caseIdx/stride, imgPixels)) + MUL24(outputPartY, imgSize) * 8 + outputPartX * 16; // hid acts for conv rbm
    targets += MUL24(caseIdx, numOutputs) + MUL24(outputPartY, numOutputsX)*8 + outputPartX*16 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    if (filterSize <= 10)
        filters += tidx;
//    filters += stride * filterPixels;

//
//
//    if(blockIdx.x != 0 || blockIdx.y != 0)
//        return;

    const float* lastFilter = filters + filterPixels * stride * numFilters; // bad pointer
    float prod = 0;
    const bool compute = !checkBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 8 + threadIdx.y < numOutputsX);
    do {
        __syncthreads();

        for (int i = tidx; i < shImgPixels; i += 16 * 8) {
            const int loadX = i % (shImgSizeX);
            const int loadY = i / (shImgSizeX);

            if (!checkBounds || (outputPartX * 16 + loadX < imgSize && outputPartY * 8 + loadY < imgSize)) {
                shImg[0][i] = imgs[MUL24(loadY, imgSize) + loadX];
            }
        }
        if (filterSize <= 10) {
            if (tidx < filterPixels)
                shFilter[0][tidx] = filters[0];
        } else {
            for (int i = tidx; i < filterPixels; i += 16 * 8) {
                const int loadX = i % (filterSize);
                const int loadY = i / (filterSize);
                shFilter[0][i] = filters[MUL24(filterSize, loadY) + loadX];
            }
        }

        __syncthreads();

        if (compute) {
            float* myShFilter = &shFilter[0][0];
            float* myShImg = &shImg[threadIdx.y][threadIdx.x];

            #pragma unroll
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    prod += myShFilter[0] * myShImg[0];

                    myShFilter++;
                    myShImg++;
                }
                myShImg += 15;
            }
        }
        imgs += imgPixels;
        filters += filterPixels * stride;
    } while (filters != lastFilter);
    if (compute)
        targets[0] = prod;
//        targets[0] = outputPartY * 16 + threadIdx.y;
}



#endif /* CONV3_CUH_ */
