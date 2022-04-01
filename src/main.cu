#define TINYOBJLOADER_IMPLEMENTATION 
#define PI 3.1415926535897932385
#define EPSILON 0.000001

#include <iostream>
#include <curand_kernel.h>  //CUDA random
#include <thrust/sort.h>

#include "aabb.h"
#include "node.h"
#include "triangle.h"
#include "vec3.h"
#include "vec2.h"
#include "raytracer/render_config.h"
#include "raytracer/ray.h"
#include "raytracer/hit.h"
#include "raytracer/camera.h"
#include "main.h"
#include "math_util.h"

#include "lbvh.cu"

#include "cuda_helpers/helper_cuda.h"      //checkCudaErrors
#include "tiny_obj_loader.h"

__global__ void render(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals){
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

__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals) {
  float cur_attenuation = 1.0f;
  for(int i = 0; i < max_depth; i++) {
    RayHit hit;
    bool was_hit = false;
    for (int j = 0; j < vertex_count/3; j++){ //TODO: Traverse acceleration structure.
      RayHit tempHit;

      if (!intersectTri(ray, &tempHit,  vertices[triangles[j].v0_index],
                                        vertices[triangles[j].v1_index],
                                        vertices[triangles[j].v2_index],
                                        normals [triangles[j].v0_index],
                                        normals [triangles[j].v1_index],
                                        normals [triangles[j].v2_index]))
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
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2){
  Vec3 edge1 = v1 - v0;
  Vec3 edge2 = v2 - v0;

  Vec3 pvec = cross(ray->direction(), edge2);
  float det = dot(edge1, pvec);
  //Culling implementation
  if(det < EPSILON)
    return false;
  
  Vec3 tvec = ray->origin() - v0;
  bestHit->uv.e[0] = dot(tvec, pvec);
  if(bestHit->uv.x() < 0.0 || bestHit->uv.x() > det)
    return false;

  Vec3 qvec = cross(tvec, edge1);
  bestHit->uv.e[1] = dot(ray->direction(), qvec);
  if(bestHit->uv.y() < 0.0 || bestHit->uv.x() + bestHit->uv.y() > det)
    return false;

  float inv_det = 1.0 / det;
  bestHit->dist = dot(edge2, qvec) * inv_det;
  bestHit->uv.e[0] *= inv_det;
  bestHit->uv.e[1] *= inv_det;
  bestHit->normal = normalize(cross(edge1, edge2));
  bestHit->pos = ray->point_along_ray(bestHit->dist);

  //BUG: There's something funky with the normals when interpolating...
  // bestHit->normal = normalize(bestHit->uv.x()*n1 + bestHit->uv.y() * n2 + (1.0 - bestHit->uv.x() - bestHit->uv.y()) * n0);
  return true;
}

//From Shirleys Ray Tracing in One Weekend.
__device__ bool intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius){
  Vec3 oc = ray->origin() - point;
  float a = sqrMagnitude(ray->direction());
  float half_b = dot(oc, ray->direction());
  float c = sqrMagnitude(oc) - radius*radius;

  float discriminant = half_b*half_b - a*c;
  if (discriminant < 0) return false;
  float sqrtd = sqrt(discriminant);

  float root = (-half_b - sqrtd) / a;
  if (root < 0.00001 || 999999.0 < root) {
      root = (-half_b + sqrtd) / a;
      if (root < 0.00001 || 999999.0 < root)
          return false;
  }
  bestHit->dist = root;
  bestHit->pos = ray->point_along_ray(bestHit->dist);
  bestHit->normal = (bestHit->pos - point) / radius;
  return true;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __host__ inline unsigned int expandBits(unsigned int v){
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

//Expects an input Vec3(0..1, 0..1, 0..1)
__device__ __host__ unsigned int mortonCode(Vec3 v){
  //Clamp coordinates to 10 bits.
  float x = min(max(v.x() * 1024.0f, 0.0f), 1023.0f);
  float y = min(max(v.y() * 1024.0f, 0.0f), 1023.0f);
  float z = min(max(v.z() * 1024.0f, 0.0f), 1023.0f);
  //Bit shift componentwise before merging bits into morton code.
  unsigned int xx = expandBits((unsigned int)x) << 2;
  unsigned int yy = expandBits((unsigned int)y) << 1;
  unsigned int zz = expandBits((unsigned int)z);
  return xx | yy | zz;
}

// https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file#C
int serializeImageBuffer(Vec3 *ptr_img, const char *file_name, int image_width, int image_height){
  FILE *fp = fopen(file_name, "w");
  fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);

  for (int j = image_height-1; j >= 0; j--) {
    for (int i = 0; i < image_width; i++) {
      size_t pixel_index = j*image_width + i;
      float r = clamp01(abs(ptr_img[pixel_index].x()));
      float g = clamp01(abs(ptr_img[pixel_index].y()));
      float b = clamp01(abs(ptr_img[pixel_index].z()));
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      fprintf(fp, "%d %d %d\n", ir, ig, ib);
    }
  }

  fclose(fp);
  return 0;
}

int main(int argc, char *argv[]){
  std::string filename = "sample_models/large_210.obj";

  for (size_t i = 2; i < argc; i+=2){
    if(strcmp(argv[i-1], "-i") == 0 || strcmp(argv[i-1], "--input") == 0)
      filename = std::string(argv[i]);
  }

  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filename, reader_config)) {
    if (!reader.Error().empty()) {
        std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
  }

  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  const tinyobj::attrib_t &attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

  std::cout << "\nFile '" << filename << "' loaded." << std::endl;
  int vertex_count = (int)(attrib.vertices.size()) / 3;
  printf("# vertices        = %d\n", vertex_count);
  int indices_count = (int)(shapes[0].mesh.indices.size());
  printf("# vertex indices  = %d\n", indices_count);
  int normals_count = (int)(attrib.normals.size()) / 3;
  printf("# normals         = %d\n\n", normals_count);


  //Calculate scene bounding box. This is not strictly required if we make sure to resolve duplicate codes.
  //Although it likely will lead to better quality structure.
  Vec3 min_bounds = Vec3( 10000000.0, 10000000.0,   10000000.0);
  Vec3 max_bounds = Vec3(-10000000.0,-10000000.0,  -10000000.0);

  for (int i = 0; i < attrib.vertices.size(); i+=3){
    float x = attrib.vertices[i  ];
    float y = attrib.vertices[i+1];
    float z = attrib.vertices[i+2];
    min_bounds.e[0] = min(min_bounds.x(), x);
    min_bounds.e[1] = min(min_bounds.y(), y);
    min_bounds.e[2] = min(min_bounds.z(), z);

    max_bounds.e[0] = max(max_bounds.x(), x);
    max_bounds.e[1] = max(max_bounds.y(), y);
    max_bounds.e[2] = max(max_bounds.z(), z);
  }
  printf("Scene bounds calculated...\n\tMin Bounds: (%f, %f, %f)\n", min_bounds.x(), min_bounds.y(), min_bounds.z());
  printf("\tMax Bounds: (%f, %f, %f)\n", max_bounds.x(), max_bounds.y(), max_bounds.z());  

  //The Obj reader does not store vertex indices in contiguous memory.
  //Copy the indices into a block of memory on the host device.
  Triangle *ptr_host_triangles = (Triangle*)malloc(sizeof(Triangle) * indices_count/3);
  for (int i = 0; i < indices_count; i+=3){
    Triangle tempTri = Triangle();
    int v0_index = shapes[0].mesh.indices[i  ].vertex_index;
    int v1_index = shapes[0].mesh.indices[i+1].vertex_index;
    int v2_index = shapes[0].mesh.indices[i+2].vertex_index;
    tempTri.v0_index = v0_index;
    tempTri.v1_index = v1_index;
    tempTri.v2_index = v2_index;

    //TODO @Perf: The morton code generation could easily be done on the GPU instead.
    tinyobj::index_t idx = shapes[0].mesh.indices[i];
    Vec3 v0 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    idx = shapes[0].mesh.indices[i+1];
    Vec3 v1 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    idx = shapes[0].mesh.indices[i+2];
    Vec3 v2 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    Vec3 centroid = (v0 + v1 + v2) / 3;

    // printf("Centroid: (%f, %f, %f)\n", centroid.x(), centroid.y(), centroid.z());           // @debug
    centroid.e[0] = (centroid.x() - min_bounds.x()) / (max_bounds.x() - min_bounds.x());
    centroid.e[1] = (centroid.y() - min_bounds.y()) / (max_bounds.y() - min_bounds.y());
    centroid.e[2] = (centroid.z() - min_bounds.z()) / (max_bounds.z() - min_bounds.z());
    // printf("Centroid: (%f, %f, %f)\n\n", centroid.x(), centroid.y(), centroid.z());         // @debug
    tempTri.morton_code = mortonCode(centroid);
    tempTri.aabb.min_bounds = min(v2, min(v0, v1)); //TODO: This should not happen here? This should be in lbvh::calculateAABB()
    tempTri.aabb.max_bounds = max(v2, max(v0, v1));
    ptr_host_triangles[i/3] = tempTri;
  }

  //Allocate and memcpy index, vertex and normal buffers from host to device.
  Triangle *ptr_device_triangles;
  checkCudaErrors(cudaMalloc((void**)&ptr_device_triangles, indices_count/3 * sizeof(Triangle)));
  checkCudaErrors(cudaMemcpy(ptr_device_triangles, ptr_host_triangles, indices_count/3 * sizeof(Triangle), cudaMemcpyHostToDevice));
  
  Vec3 *ptr_device_vertices;
  checkCudaErrors(cudaMalloc(&ptr_device_vertices, vertex_count * sizeof(Vec3)));
  checkCudaErrors(cudaMemcpy(ptr_device_vertices, attrib.vertices.data(), vertex_count * sizeof(Vec3), cudaMemcpyHostToDevice));

  Vec3 *ptr_device_normals;
  checkCudaErrors(cudaMalloc(&ptr_device_normals, normals_count * sizeof(Vec3)));
  checkCudaErrors(cudaMemcpy(ptr_device_normals, attrib.normals.data(), normals_count * sizeof(Vec3), cudaMemcpyHostToDevice));

  //Set default values for filename and image size.
  char* output_filename = "output.ppm";
                    //w    h    spp   max_depth
  RenderConfig config(512, 512, 30, 5         );
  Camera cam = Camera(config.img_width, config.img_height, 90.0f, 1.0f, Vec3(0,0,0));

  curandState *d_rand_state;
  Vec3 *ptr_device_img;
  checkCudaErrors(cudaMalloc(&d_rand_state, config.img_width * config.img_height*sizeof(curandState)));
  checkCudaErrors(cudaMalloc(&ptr_device_img, config.img_width * config.img_height*sizeof(Vec3)));

  // ----------- SORT -----------
  // Sorts the triangle buffer based on the computed morton codes. (using < overloading from the triangle struct).
  thrust::sort(thrust::device, ptr_device_triangles, ptr_device_triangles+indices_count/3);


  // ----------- CONSTRUCT Karras 2012 -----------
  int primitive_count = indices_count/3;
  BVH bvh = BVH(ptr_device_triangles, primitive_count, ptr_device_vertices, vertex_count);
  Node* ptr_device_tree = bvh.construct();
  printf("Primitives: %i, Thread blocks: %i, Threads per block: %i \n", primitive_count, primitive_count/64+1, 64);
  //TODO: Verify tree structure.


  // ----------- RENDER -----------
  int threads_x = 8;
  int threads_y = 8;
  dim3 threads(threads_x,threads_y);
  dim3 tracingBlocks(config.img_width/threads_x+1,config.img_height/threads_y+1);

  printf("Initializing kernels... ");
  initKernels<<<tracingBlocks, threads>>>(config.img_width, config.img_height, 1337, d_rand_state);
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Initialization complete.\nStarting Rendering... ");
  render<<<tracingBlocks, threads>>>(ptr_device_img, cam, d_rand_state, config, ptr_device_vertices, ptr_device_triangles, indices_count, ptr_device_normals);
  checkCudaErrors(cudaDeviceSynchronize());

  //Copy framebuffer to host and save to disk.
  Vec3 *ptr_host_img = (Vec3*)malloc(sizeof(Vec3) * config.img_height * config.img_width);
  checkCudaErrors(cudaMemcpy(ptr_host_img, ptr_device_img, sizeof(Vec3) * config.img_height * config.img_width, cudaMemcpyDeviceToHost));
  printf("Render complete.\nWriting to disk... ");
  serializeImageBuffer(ptr_host_img, output_filename, config.img_width, config.img_height);
  printf("Saved to disk.\n");

  free(ptr_host_img);
  free(ptr_host_triangles);
  checkCudaErrors(cudaFree(ptr_device_img));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(ptr_device_triangles));
  checkCudaErrors(cudaFree(ptr_device_vertices));
  checkCudaErrors(cudaFree(ptr_device_normals));
  return 0;
}