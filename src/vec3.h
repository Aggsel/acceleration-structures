#pragma once
#include "vec2.h"
#include <algorithm>

class Vec3{
  public:
    float e[3];
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float e0, float e1, float e2);
    __host__ __device__ Vec3(float e0);
    __host__ __device__ Vec3(Vec2 v2, float e2);
    __host__ __device__ float x();
    __host__ __device__ float y();
    __host__ __device__ float z();
};

__host__ __device__ Vec3 operator*(float lhs, Vec3 rhs);
__host__ __device__ Vec3 operator/(Vec3 lhs, float rhs);
__host__ __device__ Vec3 operator+(Vec3 lhs, Vec3 rhs);
__host__ __device__ Vec3 operator-(Vec3 lhs, Vec3 rhs);
__host__ __device__ float dot(Vec3 a, Vec3 b);
__host__ __device__ Vec3 abs(Vec3 a);
__host__ __device__ float magnitude(Vec3 a);
__host__ __device__ float sqrMagnitude(Vec3 a);
__host__ __device__ Vec3 cross(Vec3 a, Vec3 b);
__host__ __device__ Vec3 normalize(Vec3 v);
__host__ __device__ Vec3 min(Vec3 v1, Vec3 v2);
__host__ __device__ Vec3 max(Vec3 v1, Vec3 v2);

__host__ __device__ Vec3::Vec3(){}
__host__ __device__ Vec3::Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
__host__ __device__ Vec3::Vec3(float e0) { e[0] = e0; e[1] = e0; e[2] = e0; }
__host__ __device__ Vec3::Vec3(Vec2 v2, float e2) { e[0] = v2.x(); e[1] = v2.y(); e[2] = e2; }

__host__ __device__ float Vec3::x() { return e[0]; }
__host__ __device__ float Vec3::y() { return e[1]; }
__host__ __device__ float Vec3::z() { return e[2]; }

__host__ __device__ Vec3 operator*(float lhs, Vec3 rhs){
  return Vec3(lhs*rhs.x(), lhs*rhs.y(), lhs*rhs.z());
}
__host__ __device__ Vec3 operator/(Vec3 lhs, float rhs){
  return Vec3(lhs.x()/rhs, lhs.y()/rhs, lhs.z()/rhs);
}
__host__ __device__ Vec3 operator+(Vec3 lhs, Vec3 rhs){
  return Vec3(lhs.x()+rhs.x(), lhs.y()+rhs.y(), lhs.z()+rhs.z());
}
__host__ __device__ Vec3 operator-(Vec3 lhs, Vec3 rhs){
  return Vec3(lhs.x()-rhs.x(), lhs.y()-rhs.y(), lhs.z()-rhs.z());
}

__host__ __device__ float dot(Vec3 a, Vec3 b){
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

__host__ __device__ Vec3 abs(Vec3 a){
  return Vec3(abs(a.x()), abs(a.y()), abs(a.z()));
}

__host__ __device__ float magnitude(Vec3 a){
  return sqrt(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
}

__host__ __device__ float sqrMagnitude(Vec3 a){
  return a.x()*a.x() + a.y()*a.y() + a.z()*a.z();
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b){
  return Vec3(  (a.y() * b.z()) - (a.z() * b.y()),
                (a.z() * b.x()) - (a.x() * b.z()),
                (a.x() * b.y()) - (a.y() * b.x()));
}

__host__ __device__ Vec3 normalize(Vec3 v){
  float mag = magnitude(v);
  return Vec3(v.x()/mag, v.y()/mag, v.z()/mag);
}

__host__ __device__ Vec3 min(Vec3 v1, Vec3 v2){
  return Vec3(  min(v1.x(), v2.x()),
                min(v1.y(), v2.y()),
                min(v1.z(), v2.z()) );
}

__host__ __device__ Vec3 max(Vec3 v1, Vec3 v2){
  return Vec3(  max(v1.x(), v2.x()),
                max(v1.y(), v2.y()),
                max(v1.z(), v2.z()) );
}