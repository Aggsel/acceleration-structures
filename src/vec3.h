#pragma once

#include "vec2.h"
#include <smmintrin.h> //SSE4.1

#ifdef __CUDA_ARCH__ //DEVICE IMPLEMENTATION
class Vec3{
  public:
    float e[4];
    __device__ __host__ Vec3();
    __device__ __host__ Vec3(float e0, float e1, float e2);
    __device__ __host__ Vec3(float e0);
    __device__ __host__ Vec3(Vec2 v2, float e2);
    __device__ __host__ float x();
    __device__ __host__ float y();
    __device__ __host__ float z();
    __device__ __host__ inline float Vec3::operator[] (int i) const;
    __device__ __host__ inline float& Vec3::operator[] (int i);
};

__device__ __host__ Vec3::Vec3(){ e[0] = 0.0f; e[1] = 0.0f; e[2] = 0.0f; e[3] = 0.0f; }
__device__ __host__ Vec3::Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; e[3] = 0.0f; }
__device__ __host__ Vec3::Vec3(float e0) { e[0] = e0; e[1] = e0; e[2] = e0; e[3] = 0.0f; }
__device__ __host__ Vec3::Vec3(Vec2 v2, float e2) { e[0] = v2.x(); e[1] = v2.y(); e[2] = e2; e[3] = 0.0f; }

__device__ __host__ inline float Vec3::operator[] (int i) const { return e[i]; };
__device__ __host__ inline float& Vec3::operator[] (int i) { return e[i]; };

__device__ __host__ float Vec3::x() { return e[0]; }
__device__ __host__ float Vec3::y() { return e[1]; }
__device__ __host__ float Vec3::z() { return e[2]; }

__device__ __host__ Vec3 operator+(Vec3 lhs, Vec3 rhs){
  return Vec3(lhs.x()+rhs.x(), lhs.y()+rhs.y(), lhs.z()+rhs.z());
}
__device__ __host__ Vec3 operator-(Vec3 lhs, Vec3 rhs){
  return Vec3(lhs.x()-rhs.x(), lhs.y()-rhs.y(), lhs.z()-rhs.z());
}
__device__ __host__ Vec3 min(Vec3 v1, Vec3 v2){
  return Vec3(  min(v1.x(), v2.x()),
                min(v1.y(), v2.y()),
                min(v1.z(), v2.z()) );
}
__device__ __host__ Vec3 max(Vec3 v1, Vec3 v2){
  return Vec3(  max(v1.x(), v2.x()),
                max(v1.y(), v2.y()),
                max(v1.z(), v2.z()) );
}


#else //HOST IMPLEMENTATION
class Vec3{
  public:
    __m128 e;
    __host__ Vec3();
    __host__ Vec3(float e0, float e1, float e2);
    __host__ Vec3(float e0);
    __host__ Vec3(Vec2 v2, float e2);
    __host__ float x();
    __host__ float y();
    __host__ float z();
    __host__ inline float Vec3::operator[] (int i) const;
    __host__ inline float& Vec3::operator[] (int i);
};

// https://jdelezenne.github.io/Codex/Core/SIMD.html
__host__ Vec3::Vec3(){ e = _mm_setr_ps(0.0f, 0.0f, 0.0f, 0.0f); }
__host__ Vec3::Vec3(float e0, float e1, float e2) { e = _mm_setr_ps(e0, e1, e2, 0.0f); }
__host__ Vec3::Vec3(float e0) { e = _mm_setr_ps(e0, e0, e0, 0.0f); }
__host__ Vec3::Vec3(Vec2 v2, float e2) { e = _mm_setr_ps(v2.x(), v2.y(), e2, 0.0f); }

__host__ inline float Vec3::operator[] (int i) const { return e.m128_f32[i]; };
__host__ inline float& Vec3::operator[] (int i) { return e.m128_f32[i]; };

__host__ float Vec3::x() { return _mm_cvtss_f32(e); }
__host__ float Vec3::y() { return _mm_cvtss_f32(_mm_shuffle_ps(e, e, _MM_SHUFFLE(1, 1, 1, 1))); }
__host__ float Vec3::z() { return _mm_cvtss_f32(_mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 2, 2, 2))); }
__host__ Vec3 operator+ (Vec3 lhs, Vec3 rhs) { lhs.e = _mm_add_ps(lhs.e, rhs.e); return lhs; }
__host__ Vec3 operator- (Vec3 lhs, Vec3 rhs) { lhs.e = _mm_sub_ps(lhs.e, rhs.e); return lhs; }
__host__ Vec3 min(Vec3 lhs, Vec3 rhs){ lhs.e = _mm_min_ps(lhs.e, rhs.e); return lhs; }
__host__ Vec3 max(Vec3 lhs, Vec3 rhs){ lhs.e = _mm_max_ps(lhs.e, rhs.e); return lhs; }
#endif

// HOST AND DEVICE COMMON IMPLEMENTATION
// (implementation must be data type agnostic)
__device__ __host__ Vec3 operator*(float lhs, Vec3 rhs);
__device__ __host__ Vec3 operator/(Vec3 lhs, float rhs);
__device__ __host__ bool operator==(Vec3 lhs, Vec3 rhs);
__device__ __host__ float dot(Vec3 a, Vec3 b);
__device__ __host__ Vec3 abs(Vec3 a);
__device__ __host__ float magnitude(Vec3 a);
__device__ __host__ float sqrMagnitude(Vec3 a);
__device__ __host__ Vec3 cross(Vec3 a, Vec3 b);
__device__ __host__ Vec3 normalize(Vec3 v);
__device__ __host__ Vec3 min(Vec3 v1, Vec3 v2);
__device__ __host__ Vec3 max(Vec3 v1, Vec3 v2);
__device__ __host__ Vec3 lerp(Vec3 v1, Vec3 v2, float t);

__device__ __host__ Vec3 operator*(float lhs, Vec3 rhs){
  return Vec3(lhs*rhs.x(), lhs*rhs.y(), lhs*rhs.z());
}
__device__ __host__ Vec3 operator/(Vec3 lhs, float rhs){
  return Vec3(lhs.x()/rhs, lhs.y()/rhs, lhs.z()/rhs);
}
__device__ __host__ Vec3 operator/(float lhs, Vec3 rhs){
  return Vec3(lhs/rhs.x(), lhs/rhs.y(), lhs/rhs.z());
}
__device__ __host__ bool operator==(Vec3 lhs, Vec3 rhs){
  return (abs(lhs.x() - rhs.x()) <= FLT_EPSILON) && (abs(lhs.y() - rhs.y()) <= FLT_EPSILON) && (abs(lhs.z() - rhs.z()) <= FLT_EPSILON);
}

__device__ __host__ float dot(Vec3 a, Vec3 b){
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

__device__ __host__ Vec3 abs(Vec3 a){
  return Vec3(abs(a.x()), abs(a.y()), abs(a.z()));
}

__device__ __host__ float magnitude(Vec3 a){
  return sqrt(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
}

__device__ __host__ float sqrMagnitude(Vec3 a){
  return a.x()*a.x() + a.y()*a.y() + a.z()*a.z();
}

__device__ __host__ Vec3 cross(Vec3 a, Vec3 b){
  return Vec3(  (a.y() * b.z()) - (a.z() * b.y()),
                (a.z() * b.x()) - (a.x() * b.z()),
                (a.x() * b.y()) - (a.y() * b.x()));
}

__device__ __host__ Vec3 normalize(Vec3 v){
  float mag = magnitude(v);
  return Vec3(v.x()/mag, v.y()/mag, v.z()/mag);
}