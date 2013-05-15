/*
 * convCPU.cpp
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include "convCPU.h"

inline float dotCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x) {
    float result = 0;
    for(int fY = 0; fY < filterSize; fY++) {
        for(int fX = 0; fX < filterSize; fX++) {
            result += img[(y+fY) * imgSize + fX + x] * filter[fY * filterSize + fX];
        }
    }
    return result;
}

void convCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int f = 0; f < numFiltersPerGroup; f++) {
            for(int i = 0; i < numImgsPerGroup; i++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] = dotCPU(&imgs[i*imgPixels], &filters[f*filterPixels], imgSize, filterSize, y, x);
                    }
                }
                targets += numOutputs;
            }
        }
        filters += numFiltersPerGroup * filterPixels;
        imgs += imgPixels * numImgsPerGroup;
    }
}

void convCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i <  numImgsPerGroup; i++) {
        for(int g = 0; g < numGroups; g++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] = dotCPU(&imgs[g * numImgsPerGroup * imgPixels + i*imgPixels],
                                                                    &filters[g * numFiltersPerGroup * filterPixels + f*filterPixels],
                                                                    imgSize, filterSize, y, x);
                    }
                }
                targets += numOutputs;
            }
        }
//        filters += numFiltersPerGroup * filterPixels;
//        imgs += imgPixels * numImgsPerGroup;
    }
}

void convColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int f = 0; f < numFiltersPerGroup; f++) {
            for(int i = 0; i < numImgsPerGroup; i++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] = dotCPU(&imgs[i*3*imgPixels], &filters[f*3*filterPixels], imgSize, filterSize, y, x)
                                                     + dotCPU(&imgs[(i*3+1)*imgPixels], &filters[(f*3+1)*filterPixels], imgSize, filterSize, y, x)
                                                     + dotCPU(&imgs[(i*3+2)*imgPixels], &filters[(f*3+2)*filterPixels], imgSize, filterSize, y, x);
                    }
                }
                targets += numOutputs;
            }
        }
        filters += 3*filterPixels*numFiltersPerGroup;
        imgs += 3*imgPixels * numImgsPerGroup;
    }
}

void convColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i <  numImgsPerGroup; i++) {
        for(int g = 0; g < numGroups; g++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] = dotCPU(&imgs[g * numImgsPerGroup * 3 * imgPixels + i*3*imgPixels],
                                                              &filters[g * numFiltersPerGroup * filterPixels * 3 + f*3*filterPixels],
                                                              imgSize, filterSize, y, x)
                                                     + dotCPU(&imgs[g * numImgsPerGroup * 3 * imgPixels + (i*3+1)*imgPixels],
                                                              &filters[g * numFiltersPerGroup * filterPixels * 3 + (f*3+1)*filterPixels],
                                                              imgSize, filterSize, y, x)
                                                     + dotCPU(&imgs[g * numImgsPerGroup * 3 * imgPixels + (i*3+2)*imgPixels],
                                                              &filters[g * numFiltersPerGroup * filterPixels * 3 + (f*3+2)*filterPixels],
                                                              imgSize, filterSize, y, x);
                    }
                }
                targets += numOutputs;
            }
        }
//        filters += 3*filterPixels*numFiltersPerGroup;
//        imgs += 3*imgPixels * numImgsPerGroup;
    }
}

void conv2CPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int i = 0; i < numImgsPerGroup; i++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[g * numFiltersPerGroup * numOutputs
                                + i * numGroups * numFiltersPerGroup * numOutputs
                                + f * numOutputs + y * numOutputsX + x] = dotCPU(&imgs[i*imgPixels],
                                                                                 &filters[f * numImgsPerGroup * filterPixels
                                                                                          + i * filterPixels], imgSize, filterSize, y, x);
                    }
                }
            }
        }
        filters += filterPixels * numFiltersPerGroup * numImgsPerGroup;
        imgs += imgPixels * numImgsPerGroup;
    }
}

void conv2CPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int i = 0; i < numImgsPerGroup; i++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[g * numFiltersPerGroup * numOutputs
                                + i * numGroups * numFiltersPerGroup * numOutputs
                                + f * numOutputs + y * numOutputsX + x] = dotCPU(&imgs[i*imgPixels],
                                                                                 &filters[i * numGroups * numFiltersPerGroup * filterPixels]
                                                                                          + g * numFiltersPerGroup * filterPixels
                                                                                          + f * filterPixels, imgSize, filterSize, y, x);
                    }
                }
            }
        }
        imgs += imgPixels * numImgsPerGroup;
    }
}

void conv2ColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int i = 0; i < numImgsPerGroup; i++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[g * numFiltersPerGroup * numOutputs * 3
                                + (i/3) * numGroups * numFiltersPerGroup * numOutputs * 3
                                + f * numOutputs * 3
                                + (i%3) * numOutputs
                                + y * numOutputsX + x] = dotCPU(&imgs[i*imgPixels],
                                                                &filters[f * (numImgsPerGroup/3) * filterPixels + (i/3)*filterPixels],
                                                                imgSize, filterSize, y, x);
                    }
                }
            }
        }
        filters += filterPixels * numFiltersPerGroup * (numImgsPerGroup/3);
        imgs += imgPixels * numImgsPerGroup;
    }
}

void conv2ColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g <  numGroups; g++) {
        for(int i = 0; i < numImgsPerGroup; i++) {
            for(int f = 0; f < numFiltersPerGroup; f++) {
                for(int y = 0; y < numOutputsX; y++) {
                    for(int x = 0; x < numOutputsX; x++) {
                        targets[g * numFiltersPerGroup * numOutputs * 3
                                + (i/3) * numGroups * numFiltersPerGroup * numOutputs * 3
                                + f * numOutputs * 3
                                + (i%3) * numOutputs
                                + y * numOutputsX + x] = dotCPU(&imgs[i*imgPixels],
                                                                &filters[(i/3)*numGroups * numFiltersPerGroup * filterPixels
                                                                         + g * numFiltersPerGroup * filterPixels
                                                                         + f * filterPixels],
                                                                imgSize, filterSize, y, x);
                    }
                }
            }
        }
//        filters += filterPixels * numFiltersPerGroup * (numImgsPerGroup/3);
        imgs += imgPixels * numImgsPerGroup;
    }
}

void rotate180CPU(float* filters, float* targets, int filterSize, int numFilters) {
    int filterPixels = filterSize * filterSize;
    for(int f = 0; f < numFilters; f++) {
        for(int y = 0; y < filterSize; y++) {
            for(int x = 0; x < filterSize; x++) {
                targets[f * filterPixels + (filterSize  - 1 - y) * filterSize + filterSize -1 - x] = filters[f * filterPixels + y * filterSize + x];
            }
        }
    }
}

inline float dotRotateCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x) {
    img += y * imgSize;
    float result = 0;
    for (int fY = 0; fY < filterSize; fY++) {
        for (int fX = 0; fX < filterSize; fX++) {
            result += img[fX + x] * filter[(filterSize - 1 - fY) * filterSize + filterSize - 1 - fX];
        }
        img += imgSize;
    }
    return result;
}

void conv3CPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g < numGroups; g++) {
        for (int i = 0; i < numImgsPerGroup; i++) {
            for (int f = 0; f < numFiltersPerGroup; f++) {
                for (int y = 0; y < numOutputsX; y++) {
                    for (int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] += dotRotateCPU(&imgs[f * imgPixels * numImgsPerGroup + imgPixels * i],
                                                                     &filters[f * filterPixels], imgSize, filterSize, y, x);
                    }
                }
            }
            targets += numOutputs;
        }
        imgs += imgPixels * numImgsPerGroup * numFiltersPerGroup;
        filters += filterPixels * numFiltersPerGroup;
    }
}

void conv3CPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g < numGroups; g++) {
        for (int i = 0; i < numImgsPerGroup; i++) {
            for (int f = 0; f < numFiltersPerGroup; f++) {
                for (int y = 0; y < numOutputsX; y++) {
                    for (int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x] += dotRotateCPU(&imgs[i * numGroups * numFiltersPerGroup * imgPixels
                                                                           + g * numFiltersPerGroup * imgPixels
                                                                           + f * imgPixels],
                                                                     &filters[f * filterPixels], imgSize, filterSize, y, x);
                    }
                }
            }
            targets += numOutputs;
        }
        filters += filterPixels * numFiltersPerGroup;
    }
}

void conv3ColorCPU_gfi(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g < numGroups; g++) {
        for (int i = 0; i < numImgsPerGroup; i++) {
            for (int f = 0; f < numFiltersPerGroup; f++) {
                for (int y = 0; y < numOutputsX; y++) {
                    for (int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x
                                + (f % 3) * numOutputs] += dotRotateCPU(&imgs[(f / 3) * imgPixels * numImgsPerGroup + imgPixels * i],
                                                                        &filters[f * filterPixels], imgSize, filterSize, y, x);
                    }
                }
            }

            targets += 3 * numOutputs;
        }
        imgs += imgPixels * numImgsPerGroup * (numFiltersPerGroup / 3);
        filters += filterPixels * numFiltersPerGroup;
    }
}

void conv3ColorCPU_igf(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgsPerGroup, int numFiltersPerGroup, int numGroups) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int g = 0; g < numGroups; g++) {
        for (int i = 0; i < numImgsPerGroup; i++) {
            for (int f = 0; f < numFiltersPerGroup; f++) {
                for (int y = 0; y < numOutputsX; y++) {
                    for (int x = 0; x < numOutputsX; x++) {
                        targets[y * numOutputsX + x
                                + (f % 3) * numOutputs] += dotRotateCPU(&imgs[i * numGroups * (numFiltersPerGroup/3) * imgPixels
                                                                              + g * (numFiltersPerGroup/3) * imgPixels
                                                                              + (f/3) * imgPixels],
                                                                        &filters[f * filterPixels], imgSize, filterSize, y, x);
                    }
                }
            }

            targets += 3 * numOutputs;
        }
//        imgs += imgPixels * numImgsPerGroup * (numFiltersPerGroup / 3);
        filters += filterPixels * numFiltersPerGroup;
    }
}

inline float sumCPU(float* image, int imgSize, int factor) {
    float sum = 0;
    for(int y = 0; y < factor; y++) {
        for(int x = 0; x < factor; x++) {
            sum += image[imgSize * y + x];
        }
    }
    return sum;
}

void subsampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    int numRegions = imgSize / factor;
    float divisor = factor * factor;
    for(int i = 0; i < numImgs; i++) {
        for(int y = 0; y < numRegions; y++) {
            for(int x = 0; x < numRegions; x++) {
                targets[0] = sumCPU(&images[y * imgSize * factor + x * factor], imgSize, factor) / divisor;
                targets++;
            }
        }
        images += imgSize * imgSize;
    }
}

void supersampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    int targetSize = imgSize * factor;

    for (int i = 0; i < numImgs; i++) {
        for (int y = 0; y < targetSize; y++) {
            for (int x = 0; x < targetSize; x++) {
                targets[0] = images[(y / factor) * imgSize + x / factor];
                targets++;
            }
        }
        images += imgSize * imgSize;
    }
}

void gridToMatrixCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    //    int targetSizeX = factor*factor;
    for (int i = 0; i < numImgs * (imgSize / factor); i++) {
        for (int y = 0; y < factor; y++) {
            for (int x = 0; x < imgSize; x++) {
                targets[x * factor + y] = images[0];
                images++;
            }
        }
        targets += factor * imgSize;
    }
}

void matrixToGridCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    //    int targetSizeX = factor*factor;
    for (int i = 0; i < numImgs * (imgSize / factor); i++) {
        for (int y = 0; y < factor; y++) {
            for (int x = 0; x < imgSize; x++) {
                targets[0] = images[x * factor + y];
                targets++;
            }
        }
        images += factor * imgSize;
    }
}

void sampleMultinomialCPU(float* multi, float* randoms, float* targets, int multinomials, int nomials) {
    for(int i = 0; i < multinomials; i++) {
        float sum = 0, prevSum = 0;
        float rnd = randoms[i];
        for(int x = 0; x < nomials; x++) {
            sum += multi[0];
            targets[0] = rnd >= prevSum && rnd < sum;
//            targets[0] = sum;
//            sum += multi[0];
            targets++;
            multi++;
            prevSum = sum;
        }
    }
}
