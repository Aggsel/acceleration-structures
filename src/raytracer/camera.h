#pragma once
#include "vec3.h"
#include "macros.h"

struct ALIGN(16) Camera{
  public:
    float viewport_height;
    float viewport_width;
    float vertical_fov;
    float focal_length;
    Vec3 origin;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 lower_left_corner;
    Vec3 world_dir;

    __device__ __host__  Camera::Camera(int image_width, int image_height, float vertical_fov, float focal_length, Vec3 origin){
      this->vertical_fov = vertical_fov;
      this->focal_length = focal_length;
      this->origin = origin;

      float theta = vertical_fov * 0.0174532925f;
      float h = tan(theta/2);
      float aspect_ratio = image_width / image_height;

      this->viewport_height = 2.0 * h;
      this->viewport_width = aspect_ratio * viewport_height;

      this->horizontal = Vec3(viewport_width, 0, 0);
      this->vertical = Vec3(0, viewport_height, 0);
      this->lower_left_corner = origin - this->horizontal/2 - this->vertical/2 - Vec3(0, 0, focal_length);
    }
};

