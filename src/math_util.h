#pragma once

__device__ __host__ float clamp(float value, float min, float max){
  if(value < min)
    return min;
  if(value > max)
    return max;
  return value;
}

__device__ __host__ float clamp01(float value){
  return clamp(value, 0, 1);
}

__device__ __host__ inline float remap01(float value, float old_min, float old_max){
  return (value - old_min) / (old_max - old_min);
}