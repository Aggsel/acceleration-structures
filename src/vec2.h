#pragma once

class Vec2{
  float e[2];
  public:
    __host__ __device__ Vec2() {}
    __host__ __device__ Vec2(float e0, float e1) { e[0] = e0; e[1] = e1;}
    __host__ __device__ inline float x(){ return e[0]; }
    __host__ __device__ inline float y(){ return e[1]; }
};

__host__ __device__ Vec2 operator*(float lhs, Vec2 rhs){
  return Vec2(lhs*rhs.x(), lhs*rhs.y());
}
__host__ __device__ Vec2 operator+(Vec2 lhs, Vec2 rhs){
  return Vec2(lhs.x()+rhs.x(), lhs.y()+rhs.y());
}
__host__ __device__ Vec2 operator-(Vec2 lhs, Vec2 rhs){
  return Vec2(lhs.x()-rhs.x(), lhs.y()-rhs.y());
}