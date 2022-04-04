#pragma once
#include "third_party/cuda_helpers/helper_cuda.h"

__device__ __host__ unsigned int mortonCode(Vec3 v);
__device__ __host__ inline unsigned int expandBits(unsigned int v);