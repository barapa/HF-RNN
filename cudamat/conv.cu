/*
 * conv.cu
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <math.h>
#include <nvmatrix.cuh>
#include "conv.cuh"

void _convolve_bw(float* images, float* filters, float* targets, int numImgsPerGroup,
                  int numFiltersPerGroup, int numGroups, int imgSize, int filterSize, int stride, ORDER outputOrder) {
    assert(stride == 1 || stride == 3);
    int numOutputsX = imgSize - filterSize + 1;
//    int numOutputs = numOutputsX*numOutputsX;
    bool checkOutputBounds = numOutputsX % 16 != 0;
    if (numOutputsX <= 9) {
        /*
         * Call special dynamic routine which is fast when the number of outputs is small.
         */
        int threadsX = numOutputsX, threadsY = numOutputsX, threadsZ = 512 / (threadsX*threadsY);
        int blocksX = numImgsPerGroup * numGroups, blocksY = DIVUP(numFiltersPerGroup, threadsZ*2);
        bool checkFilterBounds = filterSize % threadsX != 0;
//        bool checkFilterIdxBounds = numFiltersPerGroup % (threadsZ*2) != 0;

        dim3 grid(blocksX, blocksY);
        dim3 threads(threadsX, threadsY, threadsZ);
        if(outputOrder == GROUP_FILTER_IMAGE) {
            if (threadsX == 2) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 2, 128, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 2, 128, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 2, 128, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 2, 128, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 3) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 3, 56, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 3, 56, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 3, 56, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 3, 56, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 4) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 4, 32, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 4, 32, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 4, 32, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 4, 32, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 5) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 5, 20, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 5, 20, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 5, 20, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 5, 20, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 6) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 6, 14, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 6, 14, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 6, 14, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 6, 14, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 7) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 7, 10, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 7, 10, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 7, 10, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 7, 10, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 8) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 8, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 8, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 8, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 8, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 9) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 9, 6, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 9, 6, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 9, 6, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 9, 6, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        } else {
            if (threadsX == 2) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 2, 128, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 2, 128, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 2, 128, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 2, 128, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 3) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 3, 56, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 3, 56, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 3, 56, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 3, 56, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 4) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 4, 32, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 4, 32, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 4, 32, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 4, 32, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 5) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 5, 20, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 5, 20, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 5, 20, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 5, 20, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 6) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 6, 14, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 6, 14, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 6, 14, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 6, 14, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 7) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 7, 10, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 7, 10, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 7, 10, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 7, 10, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 8) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 8, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 8, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 8, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 8, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (threadsX == 9) {
                if (checkFilterBounds) {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<true, 1, 9, 6, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<true, 3, 9, 6, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_nofit_dynXYZ_2per<false, 1, 9, 6, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_dynXYZ_2per<false, 3, 9, 6, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        }
   } else if(filterSize > 20) {
        bool checkFilterBounds = filterSize % 16 != 0;
        int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);
        if(outputOrder == GROUP_FILTER_IMAGE) {
            if(threadsZ == 8) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(threadsZ == 4) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(threadsZ == 2) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        } else {
            if(threadsZ == 8) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(threadsZ == 4) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(threadsZ == 2) {
                if(checkFilterBounds) {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<true, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<true, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                } else {
                    if(stride == 1) {
                        conv_bw_nofit_4x16_2per<false, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_nofit_4x16_2per<false, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        }
    } else if (filterSize > 14) {
        int threadsZ = numFiltersPerGroup >= 8 ? 8 : numFiltersPerGroup >= 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);
        if (outputOrder == GROUP_FILTER_IMAGE) {
            if(filterSize == 15) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 16) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 17) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 18) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 19) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 20) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        } else {
            if(filterSize == 15) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<15, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<15, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 16) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<16, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<16, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 17) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<17, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<17, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 18) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<18, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<18, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 19) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<19, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<19, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if(filterSize == 20) {
                if(threadsZ == 8) {
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 4){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if(threadsZ == 2){
                    if(stride == 1) {
                        conv_bw_fit_4x16_1per<20, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_1per<20, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        }
    } else {

        int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);
//            printf("numFiltersPerGroup: %d, numImgsPerGroup: %d, numGroups: %d\n", numFiltersPerGroup, numImgsPerGroup, numGroups);
        if(outputOrder == GROUP_FILTER_IMAGE) {
            if (filterSize == 1) {
                throw "try multByScalar";
            } else if (filterSize == 2) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 3) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 4) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 5) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 6) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 7) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 8) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 9) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 10) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 11) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 12) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 13) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 14) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 8, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 4, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 2, false, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        } else { // output order: image, group, filter
            if (filterSize == 1) {
                throw "try multByScalar";
            } else if (filterSize == 2) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 3) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 4) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 5) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 6) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 7) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 8) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 9) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 10) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 11) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 12) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 13) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 14) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 8, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 4, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 2, false, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void convolve(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int numGroups, bool color, ORDER outputOrder) {
    int colorMult = color ? 3 : 1;
    assert(images->getNumCols() % colorMult == 0);
    assert(filters->getNumCols() % colorMult == 0);

    //printf("images->getNumCols()=%d, images->getNumRows()=%d\n", images->getNumCols(), images->getNumRows());
    //printf("filters->getNumCols()=%d, filters->getNumRows()=%d\n", filters->getNumCols(), filters->getNumRows());



    double dImgSize = sqrt(images->getNumCols() / colorMult);
    double dFilterSize = sqrt(filters->getNumCols() / colorMult);
    assert(dImgSize == floor(dImgSize));
    assert(dFilterSize == floor(dFilterSize));
    assert(images->getNumRows() % numGroups == 0);
    assert(filters->getNumRows() % numGroups == 0);
    int imgSize = int(dImgSize);
    int filterSize = int(dFilterSize);
    int numImgsPerGroup = images->getNumRows() / numGroups;
    int numFiltersPerGroup = filters->getNumRows() / numGroups;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;

    assert(numFiltersPerGroup % 2 == 0);
    assert(targets->getNumElements() == numOutputs * numFiltersPerGroup * numImgsPerGroup * numGroups);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    if(!color) {
        _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 1, outputOrder);
    } else {
        targets->apply(NVMatrix::ZERO);
        _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3, outputOrder);
        _convolve_bw(images->getDevData() + imgPixels, filters->getDevData() + filterPixels, targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3, outputOrder);
        _convolve_bw(images->getDevData() + 2*imgPixels, filters->getDevData() + 2*filterPixels, targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3, outputOrder);
    }
}
