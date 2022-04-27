#pragma once

#include <curand_kernel.h>  //CUDA random
#include "vec3.h"
#include "math_util.h"
#include "raytracer/render_config.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void d_render(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__global__ void d_render(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals, Node* bvh_root);
__global__ void d_render_heatmap(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals, Node* bvh_root);
__global__ void initKernels(int image_width, int image_height, curandState *rand);
__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, int vertex_count, Vec3 *normals, Node* bvh_root);
__device__ float color_heatmap(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, int vertex_count, Vec3 *normals, Node* bvh_root);
__device__ Vec3 randomInUnitSphere(curandState *rand);
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2);

__global__ void normalize_output_image(Vec3 *output_image, RenderConfig config){
  int img_size = config.img_height*config.img_width;
  float max_value = -1.0f;
  float min_value = FLT_MAX;
  float avg = 0.0f;
  for (int i = 0; i < img_size; i++){
    max_value = output_image[i].x() > max_value ? output_image[i].x() : max_value;
    min_value = output_image[i].x() < min_value ? output_image[i].x() : min_value;
    avg += output_image[i].x();
  }
  avg /= img_size;

  printf("Maximum Traversed Steps: %f\n", max_value);
  printf("Minimum Traversed Steps: %f\n", min_value);
  printf("Average Traversed Steps: %f\n", avg);

  for (int i = 0; i < img_size; i++){
    float val = output_image[i].x() / max_value;
    output_image[i] = Vec3(val, val, val);
  }
}

//TODO: Reduce these rendering functions to just one?
__global__ void d_render_heatmap(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals, Node* bvh_root){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= config.img_width) || (pixel_y >= config.img_height)) return;
  int pixel_index = pixel_y*config.img_width + pixel_x;

  curandState local_rand = rand[pixel_index];
  Vec2 uv = Vec2((pixel_x + 0.2) / (config.img_width-1), (pixel_y+ 0.2) / (config.img_height-1));
  Ray ray = Ray(Vec3(0,0,0), normalize(cam.lower_left_corner + uv.x()*cam.horizontal + uv.y()*cam.vertical - Vec3(0,0,0)) );
  float out_col = color_heatmap(&ray, &local_rand, config.max_bounces, vertices, vertex_count, normals, bvh_root);

  output_image[pixel_index] = Vec3(out_col, out_col, out_col);
}

__global__ void d_render(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals, Node* bvh_root){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= config.img_width) || (pixel_y >= config.img_height)) return;
  int pixel_index = pixel_y*config.img_width + pixel_x;

  curandState local_rand = rand[pixel_index];

  Vec3 result = Vec3(0.0, 0.0, 0.0);
  for (int i = 0; i < config.samples_per_pixel; i++){
    Vec2 uv = Vec2((pixel_x + curand_uniform(&local_rand)) / (config.img_width-1), (pixel_y+ curand_uniform(&local_rand)) / (config.img_height-1));
    Ray ray = Ray(Vec3(0,0,0), normalize(cam.lower_left_corner + uv.x()*cam.horizontal + uv.y()*cam.vertical - Vec3(0,0,0)) );
    Vec3 out_col = color(&ray, &local_rand, config.max_bounces, vertices, vertex_count, normals, bvh_root);

    float r = clamp01(out_col.x());
    float g = clamp01(out_col.y());
    float b = clamp01(out_col.z());
    result = result + Vec3(r,g,b);
  }
  
  //Gamma correction
  float scale = 1.0 / config.samples_per_pixel;
  float r = sqrt(scale * result.x());
  float g = sqrt(scale * result.y());
  float b = sqrt(scale * result.z());
  result = Vec3(r,g,b);

  output_image[pixel_index] = result;
}

__global__ void d_render(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= config.img_width) || (pixel_y >= config.img_height)) return;
  int pixel_index = pixel_y*config.img_width + pixel_x;

  curandState local_rand = rand[pixel_index];

  Vec3 result = Vec3(0.0, 0.0, 0.0);
  for (int i = 0; i < config.samples_per_pixel; i++){
    Vec2 uv = Vec2((pixel_x + curand_uniform(&local_rand)) / (config.img_width-1), (pixel_y+ curand_uniform(&local_rand)) / (config.img_height-1));
    Ray ray = Ray(Vec3(0,0,0), normalize(cam.lower_left_corner + uv.x()*cam.horizontal + uv.y()*cam.vertical - Vec3(0,0,0)) );
    Vec3 out_col = color(&ray, &local_rand, config.max_bounces, vertices, triangles, vertex_count, normals);

    float r = clamp01(out_col.x());
    float g = clamp01(out_col.y());
    float b = clamp01(out_col.z());
    result = result + Vec3(r,g,b);
  }
  
  //Gamma correction
  float scale = 1.0 / config.samples_per_pixel;
  float r = sqrt(scale * result.x());
  float g = sqrt(scale * result.y());
  float b = sqrt(scale * result.z());
  result = Vec3(r,g,b);

  output_image[pixel_index] = result;
}

//As mentioned in Accelerated Ray Tracing in One Weekend (https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
//It's a good idea to seperate initialization and actual rendering if we want accurate performance numbers.
__global__ void initKernels(int image_width, int image_height, unsigned long long rand_seed, curandState *rand){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= image_width) || (pixel_y >= image_height))
    return;
  int pixel_index = pixel_y*image_width + pixel_x;

  curand_init(rand_seed, pixel_index, 0, &rand[pixel_index]);
}

__device__ float color_heatmap(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, int vertex_count, Vec3 *normals, Node* bvh_root) {
  Node* stack[128];
  int stack_index = -1;
  stack_index++;
  stack[stack_index] = nullptr;

  int nodes_traversed = 0;
  Node* node = bvh_root;

  RayHit hit;
  do{
    Node* left_child = node->left_child;
    Node* right_child = node->right_child;

    bool left_aabb_intersect = left_child->aabb.intersectRay(*ray);
    bool right_aabb_intersect = right_child->aabb.intersectRay(*ray);

    if(left_aabb_intersect && left_child->is_leaf){
      Triangle leaf_primitive = *left_child->primitive;
      int v0 = leaf_primitive.v0_index;
      int v1 = leaf_primitive.v1_index;
      int v2 = leaf_primitive.v2_index;

      RayHit temp_hit;
      if (intersectTri(ray, &temp_hit, vertices[v0], vertices[v1], vertices[v2],
                                      normals [v0], normals [v1], normals [v2])){
        if(temp_hit.dist < hit.dist){
          hit.dist = temp_hit.dist;
          hit.normal = temp_hit.normal;
          hit.pos = temp_hit.pos;
          hit.uv = temp_hit.uv;
        }
      }
    }
    if(right_aabb_intersect && right_child->is_leaf){
      Triangle leaf_primitive = *right_child->primitive;
      int v0 = leaf_primitive.v0_index;
      int v1 = leaf_primitive.v1_index;
      int v2 = leaf_primitive.v2_index;

      RayHit temp_hit;
      if (intersectTri(ray, &temp_hit, vertices[v0], vertices[v1], vertices[v2],
                                      normals [v0], normals [v1], normals [v2])){
        if(temp_hit.dist < hit.dist){
          hit.dist = temp_hit.dist;
          hit.normal = temp_hit.normal;
          hit.pos = temp_hit.pos;
          hit.uv = temp_hit.uv;
        }
      }
    }

    bool traverse_left = (!left_child->is_leaf) && left_aabb_intersect;
    bool traverse_right = (!right_child->is_leaf) && right_aabb_intersect;

    if(!traverse_left && !traverse_right){
      node = stack[stack_index];
      stack_index--;
    }
    else{
      //Prioritize traversing left branch.
      node = traverse_left ? left_child : right_child;

      //Push right child onto the stack if both branches should be traversed.
      if (traverse_left && traverse_right){
        stack_index++;
        stack[stack_index] = right_child;
      }
    }
    nodes_traversed+=1.0;
  }while(node != nullptr);

  return nodes_traversed;
}

// Traversal: https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, int vertex_count, Vec3 *normals, Node* bvh_root) {
  float cur_attenuation = 1.0f;
  
  for(int i = 0; i < max_depth; i++) {
    Node* stack[128];
    int stack_index = -1;
    stack_index++;
    stack[stack_index] = nullptr;

    int nodes_traversed = 0;
    Node* node = bvh_root;

    bool was_hit = false;
    RayHit hit;

    do{
      Node* left_child = node->left_child;
      Node* right_child = node->right_child;

      bool left_aabb_intersect = left_child->aabb.intersectRay(*ray);
      bool right_aabb_intersect = right_child->aabb.intersectRay(*ray);

      if(left_aabb_intersect && left_child->is_leaf){
        Triangle leaf_primitive = *left_child->primitive;
        int v0 = leaf_primitive.v0_index;
        int v1 = leaf_primitive.v1_index;
        int v2 = leaf_primitive.v2_index;

        RayHit temp_hit;
        if (intersectTri(ray, &temp_hit, vertices[v0], vertices[v1], vertices[v2],
                                        normals [v0], normals [v1], normals [v2])){
          if(temp_hit.dist < hit.dist){
            hit.dist = temp_hit.dist;
            hit.normal = temp_hit.normal;
            hit.pos = temp_hit.pos;
            hit.uv = temp_hit.uv;
            was_hit = true;
          }
        }
      }
      if(right_aabb_intersect && right_child->is_leaf){
        Triangle leaf_primitive = *right_child->primitive;
        int v0 = leaf_primitive.v0_index;
        int v1 = leaf_primitive.v1_index;
        int v2 = leaf_primitive.v2_index;

        RayHit temp_hit;
        if (intersectTri(ray, &temp_hit, vertices[v0], vertices[v1], vertices[v2],
                                        normals [v0], normals [v1], normals [v2])){
          if(temp_hit.dist < hit.dist){
            hit.dist = temp_hit.dist;
            hit.normal = temp_hit.normal;
            hit.pos = temp_hit.pos;
            hit.uv = temp_hit.uv;
            was_hit = true;
          }
        }
      }

      bool traverse_left = (!left_child->is_leaf) && left_aabb_intersect;
      bool traverse_right = (!right_child->is_leaf) && right_aabb_intersect;

      if(!traverse_left && !traverse_right){
        node = stack[stack_index];
        stack_index--;
      }
      else{
        //Prioritize traversing left branch.
        node = traverse_left ? left_child : right_child;

        //Push right child onto the stack if both branches should be traversed.
        if (traverse_left && traverse_right){
          stack_index++;
          stack[stack_index] = right_child;
        }
      }
      nodes_traversed++;
    }while(node != nullptr);

    if(was_hit){
      Vec3 target = hit.pos + hit.normal + randomInUnitSphere(rand);
      cur_attenuation *= 0.5f;
      ray->org = hit.pos;
      ray->dir = normalize(target - hit.pos);
      continue;
    }
    else {
      Vec3 unit_direction = normalize(ray->direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      Vec3 c = (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0);
}

__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals) {
  float cur_attenuation = 1.0f;
  for(int i = 0; i < max_depth; i++) {
    RayHit hit;
    bool was_hit = false;
    for (int j = 0; j < vertex_count/3; j++){
      RayHit tempHit;

      if (!intersectTri(ray, &tempHit,  vertices[triangles[j].v0_index], vertices[triangles[j].v1_index], vertices[triangles[j].v2_index],
                                        normals [triangles[j].v0_index], normals [triangles[j].v1_index], normals [triangles[j].v2_index]))
        continue; //Did not hit triangle.
      
      if(tempHit.dist > hit.dist)
        continue; //Hit triangle but not closest intersection so far.

      hit.dist = tempHit.dist;
      hit.normal = tempHit.normal;
      hit.pos = tempHit.pos;
      hit.uv = tempHit.uv;
      was_hit = true;
    }

    if(was_hit){
      Vec3 target = hit.pos + hit.normal;// + randomInUnitSphere(rand);
      cur_attenuation *= 0.5f;
      ray->org = hit.pos;
      ray->dir = normalize(target - hit.pos);
      continue;
    }
    else {
      Vec3 unit_direction = normalize(ray->direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      Vec3 c = (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0);
}

__device__ Vec3 randomInUnitSphere(curandState *rand){
  while(true){
    float x = (curand_uniform(rand) * 2.0) - 1.0;
    float y = (curand_uniform(rand) * 2.0) - 1.0;
    float z = (curand_uniform(rand) * 2.0) - 1.0;
    Vec3 p = Vec3(x, y, z);
    if(sqrMagnitude(p) >= 1)
      continue;
    return p;
  }
}

/* From Möller & Trumbore, Fast, Minimum Storage Ray/Triangle Intersection */
__device__ bool intersectTri(Ray *ray, RayHit *hit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2){
  Vec3 edge1 = v1 - v0;
  Vec3 edge2 = v2 - v0;

  Vec3 pvec = cross(ray->direction(), edge2);
  float det = dot(edge1, pvec);

  if(det < EPSILON)
    return false;
  
  Vec3 tvec = ray->origin() - v0;
  hit->uv.e[0] = dot(tvec, pvec);
  if(hit->uv.x() < 0.0 || hit->uv.x() > det)
    return false;

  Vec3 qvec = cross(tvec, edge1);
  hit->uv.e[1] = dot(ray->direction(), qvec);
  if(hit->uv.y() < 0.0 || hit->uv.x() + hit->uv.y() > det)
    return false;

  float inv_det = 1.0 / det;
  float dist = dot(edge2, qvec) * inv_det;
  if (dist < 0.001)
    return false;

  hit->dist = dist;
  hit->uv.e[0] *= inv_det;
  hit->uv.e[1] *= inv_det;
  hit->normal = normalize(cross(edge1, edge2));
  hit->pos = ray->point_along_ray(hit->dist);

  return true;
}

class Raytracer{
    curandState *d_rand_state;
    Vec3 *ptr_device_img;
    RenderConfig config;

    dim3 threads, blocks;

    Vec3 *ptr_device_vertices;
    Vec3 *ptr_device_normals;
    Triangle *ptr_device_triangles;
    int index_count;

    public:
    Raytracer(RenderConfig render_config, Vec3 *device_vertex_buffer, Vec3 *normal_buffer, Triangle *triangle_buffer, int index_count){
        config = render_config;
        ptr_device_vertices = device_vertex_buffer;
        ptr_device_normals = normal_buffer;
        ptr_device_triangles = triangle_buffer;
        this->index_count = index_count;

        cudaMalloc(&d_rand_state, config.img_width * config.img_height*sizeof(curandState));
        cudaMalloc(&ptr_device_img, config.img_width * config.img_height*sizeof(Vec3));

        int threads_x = 8;
        int threads_y = 8;
        threads = dim3(threads_x,threads_y);
        blocks = dim3(config.img_width/threads_x+1,config.img_height/threads_y+1);

        initKernels<<<blocks, threads>>>(config.img_width, config.img_height, config.seed, d_rand_state);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~Raytracer(){
        cudaFree(ptr_device_img);
        cudaFree(d_rand_state);
    }

    Vec3* render(Camera cam){
      d_render<<<blocks, threads>>>(ptr_device_img, cam, d_rand_state, config, ptr_device_vertices, ptr_device_triangles, index_count, ptr_device_normals);
      checkCudaErrors(cudaDeviceSynchronize());
      return ptr_device_img;
    }

    Vec3* render(Camera cam, Node* bvh_root){
      d_render<<<blocks, threads>>>(ptr_device_img, cam, d_rand_state, config, ptr_device_vertices, ptr_device_triangles, index_count, ptr_device_normals, bvh_root);
      checkCudaErrors(cudaDeviceSynchronize());
      return ptr_device_img;
    }

    Vec3* renderTraversalHeatmap(Camera cam, Node* bvh_root){
      d_render_heatmap<<<blocks, threads>>>(ptr_device_img, cam, d_rand_state, config, ptr_device_vertices, ptr_device_triangles, index_count, ptr_device_normals, bvh_root);
      checkCudaErrors(cudaDeviceSynchronize());
      normalize_output_image<<<1,1>>>(ptr_device_img, config);
      return ptr_device_img;
    }
};