/*
 * convCPU.h
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef CONVCPU_H_
#define CONVCPU_H_

inline float dotCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x);
void convCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void convColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void convCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void convColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);

void conv2CPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv2ColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv2CPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv2ColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);

void conv3CPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv3ColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv3CPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);
void conv3ColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups);

void rotate180CPU(float* filters, float* targets, int filterSize, int numFilters);

void subsampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void supersampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void gridToMatrixCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void matrixToGridCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void sampleMultinomialCPU(float* multi, float* randoms, float* targets, int multinomials, int nomials);


template<bool add>
void copyIntoCPU(float* images, float* targets, int imgSize, int numImages, int paddingSize) {
    int targetSize = imgSize + 2*paddingSize;
    for(int i = 0; i < numImages; i++) {
        targets += paddingSize * targetSize;
        for(int y = 0; y < imgSize; y++) {
            targets += paddingSize;
            for(int x = 0; x < imgSize; x++) {
                if(add) {
                    *targets += *images;
                } else {
                    *targets = *images;
                }

                images++;
                targets++;
            }
            targets += paddingSize;
        }
        targets += paddingSize * targetSize;
    }
}
template<bool add>
void copyOutOfCPU(float* images, float* targets, int imgSize, int numImages, int paddingSize) {
    int targetSize = imgSize - 2*paddingSize;
    for(int i = 0; i < numImages; i++) {
        images += paddingSize * imgSize;
        for(int y = 0; y < targetSize; y++) {
            images += paddingSize;
            for(int x = 0; x < targetSize; x++) {
                if(add) {
                    *targets += *images;
                } else {
                    *targets = *images;
                }

                images++;
                targets++;
            }
            images += paddingSize;
        }
        images += paddingSize * imgSize;
    }
}

#endif /* CONVCPU_H_ */
