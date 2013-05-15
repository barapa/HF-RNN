/*
 * testconv.cu
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>

#include "testconv_extras.cuh"
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
    cutilCheckError(cutCreateTimer( &timer));
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

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
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
    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    convolve(&nvImages, &nvFilters, &nvTargets, numGroups, color, order);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

/*
 * This tests the routines in conv2.cuh. See the documentation there for an explanation.
 */
void test_convolve2(int imgSize, int filterSize, bool color) {
    printf("===============================\n");
    printf("test_convolve2\n");
    printf("===============================\n");

    ORDER order = IMAGE_GROUP_FILTER;
    int numFiltersPerGroup = 24, numImgsPerGroup = 64, numGroups = 4;
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
    Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
    Matrix filters(order == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
                   order == GROUP_FILTER_IMAGE ? numImgsPerGroup * filterPixels : numFiltersPerGroup * filterPixels); // == targets in test_convolve
    Matrix targets(numImgsPerGroup, numGroups * numFiltersPerGroup * numOutputs * colorMult);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
    if(color) {
        if(order == GROUP_FILTER_IMAGE) {
            conv2ColorCPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup * 3, numFiltersPerGroup, numGroups);
        } else {
            conv2ColorCPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup * 3, numFiltersPerGroup, numGroups);
        }
    } else {
        if(order == GROUP_FILTER_IMAGE) {
            conv2CPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        } else {
            conv2CPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        }
    }
    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
    convolve2(&nvImages, &nvFilters, &nvTargets, filterSize, numGroups, color, order);
    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_rot180(int filterSize, bool color) {
    printf("===============================\n");
    printf("test_rot180\n");
    printf("===============================\n");

    int numFilters = 64;
    int filterPixels = filterSize * filterSize;
    int colorMult = color ? 3 : 1;
    printf("Filters: %d\n",  numFilters);
    printf("Color: yes\n");

    Matrix filters(numFilters, colorMult * filterPixels);
    Matrix targets(filters);

    filters.randomizeUniform();

    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    rotate180CPU(filters.getData(), targets.getData(), filterSize, colorMult * numFilters);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    rotate180(&nvFilters, &nvTargets, color);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_copyInto(int imgSize, int paddingSize, bool color, bool add) {
    printf("===============================\n");
    printf("test_copyInto\n");
    printf("===============================\n");

    int numImages = 128*48;
    int imgPixels = imgSize * imgSize;
    int targetSize = imgSize + 2*paddingSize;
    int targetPixels = targetSize * targetSize;

    printf("Filters: %d\n",  numImages);
    printf("Color: yes\n");

    int colorMult = color ? 3 : 1;
    Matrix images(numImages, colorMult * imgPixels);
    Matrix targets(numImages, colorMult * targetPixels);

    images.randomizeUniform();
    targets.randomizeUniform();

    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
    if(add) {
        copyIntoCPU<true>(images.getData(), targets.getData(), imgSize, colorMult * numImages, paddingSize);
    } else {
        copyIntoCPU<false>(images.getData(), targets.getData(), imgSize, colorMult * numImages, paddingSize);
    }

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    if(add) {
        addInto(&nvImages, &nvTargets, paddingSize, color);
    } else {
        copyInto(&nvImages, &nvTargets, paddingSize, color);
    }

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_copyOutOf(int imgSize, int paddingSize, bool color, bool add) {
    printf("===============================\n");
    printf("test_copyOutOf\n");
    printf("===============================\n");

    int numImages = 128*48;
    int imgPixels = imgSize * imgSize;
    int targetSize = imgSize - 2*paddingSize;
    int targetPixels = targetSize * targetSize;

    printf("Filters: %d\n",  numImages);
    printf("Color: yes\n");

    int colorMult = color ? 3 : 1;
    Matrix images(numImages, colorMult * imgPixels);
    Matrix targets(numImages, colorMult * targetPixels);

    images.randomizeUniform();

    targets.randomizeUniform();

    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    if(add) {
        copyOutOfCPU<true>(images.getData(), targets.getData(), imgSize, colorMult * numImages, paddingSize);
    } else {
        copyOutOfCPU<false>(images.getData(), targets.getData(), imgSize, colorMult * numImages, paddingSize);
    }

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    if(add) {
        addOutOf(&nvImages, &nvTargets, paddingSize, color);
    } else {
        copyOutOf(&nvImages, &nvTargets, paddingSize, color);
    }

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

/*
 * This tests the routines in conv3.cuh. See the documentation there for an explanation.
 */
void test_convolve3(int imgSize, int filterSize, bool color) {
    printf("===============================\n");
    printf("test_convolve3\n");
    printf("===============================\n");

    ORDER imgOrder = IMAGE_GROUP_FILTER;
    int numFiltersPerGroup = 48, numImgsPerGroup = 128, numGroups = 1;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
//    assert(numFiltersPerGroup % 8 == 0);
    printf("Images: %d, filters: %d\n", numImgsPerGroup, numFiltersPerGroup);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Output grid: %dx%d\n", numOutputsX, numOutputsX);
    printf("Color: %s\n", color ? "yes" : "no");

    int colorMult = color ? 3 : 1;
    Matrix images(imgOrder == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
                  imgOrder == GROUP_FILTER_IMAGE ? numImgsPerGroup * imgPixels    : numFiltersPerGroup * imgPixels);
    Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
    Matrix targets(numImgsPerGroup * numGroups, numOutputs*colorMult);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
    if(color) {
        if(imgOrder == GROUP_FILTER_IMAGE) {
            conv3ColorCPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup*3, numGroups);
        } else {
            conv3ColorCPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup*3, numGroups);
        }
    } else {
        if(imgOrder == GROUP_FILTER_IMAGE) {
            conv3CPU_gfi(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        } else {
            conv3CPU_igf(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numImgsPerGroup, numFiltersPerGroup, numGroups);
        }
    }
    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 6, 14, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));
//    images.print(0,6,63*16,6);
    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    convolve3(&nvImages, &nvFilters, &nvTargets, numGroups, color, imgOrder);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 6, 14, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_subsample(int imgSize, int factor) {
    printf("===============================\n");
    printf("test_subsample\n");
    printf("===============================\n");

    assert(imgSize > factor && imgSize % factor == 0);
    int numImages = 128 * 64;
    int imgPixels = imgSize * imgSize;
    int numRegionsPerImage = (imgSize / factor)*(imgSize / factor);

    printf("Images: %d\n",  numImages);

    Matrix images(numImages, imgPixels);
    Matrix targets(numImages, numRegionsPerImage);

    images.randomizeUniform();

    targets.apply(Matrix::ZERO);

    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    subsampleCPU(images.getData(), targets.getData(), imgSize, factor, numImages);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    subsample(&nvImages, &nvTargets, factor);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}


void test_supersample(int imgSize, int factor) {
    printf("===============================\n");
    printf("test_supersample\n");
    printf("===============================\n");

    int numImages = 384;
    int imgPixels = imgSize * imgSize;
    int targetPixels = imgPixels*factor*factor;

    printf("Images: %d\n",  numImages);
    printf("Image size: %dx%d\n", imgSize, imgSize);
    printf("Output size: %dx%d\n", imgSize*factor, imgSize*factor);

    NVMatrix nvImages(numImages, imgPixels, false);
    NVMatrix nvTargets(numImages, targetPixels, false);

    nvImages.randomizeUniform();
    nvTargets.apply(NVMatrix::ZERO);

    Matrix images(numImages, imgPixels);
    Matrix targets(numImages, targetPixels);

    nvImages.copyToHost(images);
    targets.apply(Matrix::ZERO);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    supersampleCPU(images.getData(), targets.getData(), imgSize, factor, numImages);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    supersample(&nvImages, &nvTargets, factor);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_gridToMatrix(int imgSize, int squareSize) {
    printf("===============================\n");
    printf("test_gridToMatrix\n");
    printf("===============================\n");
    assert(imgSize % squareSize == 0);
    int numImages = 128*96;
    int imgPixels = imgSize * imgSize;
    int regionsPerImage = (imgSize / squareSize) * (imgSize / squareSize);

    printf("Images: %d\n",  numImages);
    printf("Image size: %dx%d\n", imgSize, imgSize);
    printf("Square size: %dx%d\n", squareSize, squareSize);
    printf("Output matrix: %dx%d\n", numImages * squareSize * regionsPerImage, squareSize);

    NVMatrix nvImages(numImages, imgPixels, false);
    NVMatrix nvTargets(numImages * regionsPerImage, squareSize * squareSize, false);

    nvImages.randomizeUniform();
    nvTargets.apply(NVMatrix::ZERO);

    Matrix images(numImages, imgPixels);
    Matrix targets(numImages * regionsPerImage, squareSize * squareSize);

    nvImages.copyToHost(images);
    targets.apply(Matrix::ZERO);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrixCPU(images.getData(), targets.getData(), imgSize, squareSize, numImages);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrix(&nvImages, &nvTargets, squareSize, true);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}


void test_matrixToGrid(int imgSize, int squareSize) {
    printf("===============================\n");
    printf("test_matrixToGrid\n");
    printf("===============================\n");
    assert(imgSize % squareSize == 0);
    int numImages = 128*96;
    int imgPixels = imgSize * imgSize;
    int regionsPerImage = (imgSize / squareSize) * (imgSize / squareSize);

    printf("Images: %d\n",  numImages);
    printf("Image size: %dx%d\n", imgSize, imgSize);
    printf("Square size: %dx%d\n", squareSize, squareSize);
    printf("Output matrix: %dx%d\n", numImages, imgPixels);

    NVMatrix nvImages(numImages * regionsPerImage, squareSize * squareSize, false);
    NVMatrix nvTargets(numImages, imgPixels, false);

    nvImages.randomizeUniform();
    nvTargets.apply(NVMatrix::ZERO);

    Matrix images(numImages * regionsPerImage, squareSize * squareSize);
    Matrix targets(numImages, imgPixels);

    nvImages.copyToHost(images);
    targets.apply(Matrix::ZERO);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    matrixToGridCPU(images.getData(), targets.getData(), imgSize, squareSize, numImages);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    matrixToGrid(&nvImages, &nvTargets, squareSize, true);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_localMax(int imgSize, int squareSize) {
    printf("===============================\n");
    printf("test_localMax\n");
    printf("===============================\n");
    assert(imgSize % squareSize == 0);
    int numImages = 128*64;
    int imgPixels = imgSize * imgSize;
    int regionsPerImage = (imgSize / squareSize) * (imgSize / squareSize);

    NVMatrix nvImages(numImages, imgPixels, false);
    NVMatrix nvTargets(numImages * regionsPerImage, squareSize * squareSize, false);
    NVMatrix nvTargetsMax(numImages * regionsPerImage, 1, false);

    nvImages.randomizeUniform();
    nvTargets.apply(NVMatrix::ZERO);
    nvTargetsMax.apply(NVMatrix::ZERO);

    Matrix images(numImages, imgPixels);
    Matrix targets(numImages * regionsPerImage, squareSize * squareSize);
    Matrix targetsMax(numImages * regionsPerImage, 1);

    printf("Images: %d\n",  numImages);
    printf("Image size: %dx%d\n", imgSize, imgSize);
    printf("Square size: %dx%d\n", squareSize, squareSize);
    printf("Output matrix: %dx%d\n", nvTargets.getNumRows(), nvTargets.getNumCols());

    nvImages.copyToHost(images);
    targets.apply(Matrix::ZERO);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrixCPU(images.getData(), targets.getData(), imgSize, squareSize, numImages);
    targets.max(1, targetsMax);
    targets.eltWiseDivideByVector(targetsMax);
    matrixToGridCPU(targets.getData(), images.getData(), imgSize, squareSize, numImages);
    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrix(&nvImages, &nvTargets, squareSize, true);
    nvTargets.max(1, nvTargetsMax);
    nvTargets.eltWiseDivideByVector2(nvTargetsMax);
    matrixToGrid(&nvTargets, &nvImages, squareSize, true);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_localSum(int imgSize, int squareSize) {
    printf("===============================\n");
    printf("test_localSum\n");
    printf("===============================\n");
    assert(imgSize % squareSize == 0);
    int numImages = 128*64;
    int imgPixels = imgSize * imgSize;
    int regionsPerImage = (imgSize / squareSize) * (imgSize / squareSize);

    NVMatrix nvImages(numImages, imgPixels, false);
    NVMatrix nvTargets(numImages * regionsPerImage, squareSize * squareSize, false);
    NVMatrix nvTargetsSum(numImages * regionsPerImage, 1, false);

    nvImages.randomizeUniform();
    nvTargets.apply(NVMatrix::ZERO);
    nvTargetsSum.apply(NVMatrix::ZERO);

    Matrix images(numImages, imgPixels);
    Matrix targets(numImages * regionsPerImage, squareSize * squareSize);
    Matrix targetsSum(numImages * regionsPerImage, 1);

    printf("Images: %d\n",  numImages);
    printf("Image size: %dx%d\n", imgSize, imgSize);
    printf("Square size: %dx%d\n", squareSize, squareSize);
    printf("Output matrix: %dx%d\n", nvTargets.getNumRows(), nvTargets.getNumCols());

    nvImages.copyToHost(images);
    targets.apply(Matrix::ZERO);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrixCPU(images.getData(), targets.getData(), imgSize, squareSize, numImages);
    targets.sum(1, targetsSum);
    targets.eltWiseDivideByVector(targetsSum);
    matrixToGridCPU(targets.getData(), images.getData(), imgSize, squareSize, numImages);
    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 10);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    gridToMatrix(&nvImages, &nvTargets, squareSize, true);
    nvTargets.sum(1, nvTargetsSum);
    nvTargets.eltWiseDivideByVector2(nvTargetsSum);
    matrixToGrid(&nvTargets, &nvImages, squareSize, true);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 10);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}

void test_sampleMultinomial(int nomials) {
    printf("===============================\n");
    printf("test_sampleMultinomial\n");
    printf("===============================\n");
    int multinomials = 128*25*96;

    Matrix multi(multinomials, nomials);
    Matrix randoms(multinomials,1);
    Matrix targets(multi);

    multi.randomizeUniform();
    Matrix& multiSum = multi.sum(1);
    multiSum.addScalar(1); // this will make "none of the above" an option
    multi.eltWiseDivideByVector(multiSum);
//    multi.print(3,16);
    randoms.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvMulti(multi, true);
    NVMatrix nvRandoms(randoms, true);
    NVMatrix nvTargets(targets, true);

    printf("Multinomial distributions: %d\n",  multinomials);
    printf("Multinomial distribution size: %d\n", nomials);

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));

    sampleMultinomialCPU(multi.getData(), randoms.getData(), targets.getData(),multinomials, nomials);

    cutilCheckError(cutStopTimer(timer));
    printf("CPU (partial) result:\n");
    targets.print(32*16-3, 6, 0, 8);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    cutilCheckError(cutResetTimer(timer));
    cutilCheckError(cutStartTimer(timer));
//    nvMulti.print(3,16);
    sampleMultinomial(&nvMulti, &nvRandoms, &nvTargets);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(32*16-3, 6, 0, 8);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);

    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    Matrix &s = cpuNVTargets.sum(0);
    printf("Number of distributions sampled differently: %d (this may be non-zero, but only slightly)\n", int(s.sum())/2);
}

int main(int argc, char** argv) {
    // This line just for compiling and examining profiler output.
//    exit(0); conv2_bw_nofit_dynXYZ_2per<true, false,3,8,8><<<1,1>>>(NULL, NULL, NULL, 0,0, 0);
//    exit(0); conv_bw_fit_4x16_2per<9,3,8, false><<<1,1>>>(NULL, NULL, NULL, 0, 0, 0);
//    exit(0); kSampleSmallMultinomial<15,16><<<1,1>>>(NULL, NULL, NULL, 0, 0);
//    exit(0); conv3_bw_fit_16x16<9, true,1><<<1,1>>>(NULL, NULL, NULL, 0, 0, 0);
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
    test_convolve2(32, 25, true);
    test_convolve3(60, 45, true);

    test_rot180(7, true);

    test_copyInto(24, 8, false, true);
    test_copyOutOf(24, 3, true, true);

    test_subsample(32, 4);
    test_supersample(8, 3);

    test_gridToMatrix(25, 5);
    test_matrixToGrid(25, 5);

    test_localMax(32, 4); // I dont' really use them.
    test_localSum(32, 4); // They are an example of how to do things with matrix-to-grid. 

    test_sampleMultinomial(49);
}
