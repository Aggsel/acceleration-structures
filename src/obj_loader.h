#pragma once

#include <iostream>
#include <limits>
#include "third_party/tiny_obj_loader.h"
#include "vec3.h"
#include "aabb.h"
#include "triangle.h"

class ObjLoader{
	std::string filename;
	tinyobj::ObjReaderConfig reader_config;
	tinyobj::ObjReader reader;
	tinyobj::attrib_t attrib;

	AABB scene_bounding_box;
	Triangle *ptr_host_triangles;

	public:
	int vertex_count;
	int index_count;
	int normals_count;
	int triangle_count;

	ObjLoader(std::string filename){
		if (!reader.ParseFromFile(filename, reader_config)) {
			if (!reader.Error().empty()) {
				std::cerr << "TinyObjReader: " << reader.Error();
			}
			exit(1);
		}

		if (!reader.Warning().empty()) {
			std::cout << "TinyObjReader: " << reader.Warning();
		}

		attrib = reader.GetAttrib();
		const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

		vertex_count = (int)(attrib.vertices.size()) / 3;
		index_count = (int)(shapes[0].mesh.indices.size());
		normals_count = (int)(attrib.normals.size()) / 3;
		triangle_count = index_count / 3;

		printf("\nFile %s loaded.\n", filename.c_str());
		printf("\t# vertices        = %d\n",   vertex_count);
		printf("\t# vertex indices  = %d\n",   index_count);
		printf("\t# normals         = %d\n",   normals_count);
		printf("\t# triangles       = %d\n\n", triangle_count);

		// ------------ Scene bounding box -----------------
		Vec3 min_bounds(FLT_MAX, FLT_MAX, FLT_MAX);
		Vec3 max_bounds(-FLT_MAX, -FLT_MAX, -FLT_MAX);

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
		scene_bounding_box = AABB::AABB(min_bounds, max_bounds);
		Vec3 size = max_bounds - min_bounds;
		printf("Scene bounds calculated...\n\tMin Bounds: (%f, %f, %f)\n", min_bounds.x(), min_bounds.y(), min_bounds.z());
		printf("\tMax Bounds: (%f, %f, %f)\n", max_bounds.x(), max_bounds.y(), max_bounds.z());
		printf("\tSize: (%f, %f, %f)\n", size.x(), size.y(), size.z());

		//The Obj reader does not store vertex indices in contiguous memory.
		//Copy the indices into a block of memory on the host device.
		//This is required beforehand regardless och BVH construction method.
		ptr_host_triangles = (Triangle*)malloc(sizeof(Triangle) * triangle_count);
		for (int i = 0; i < index_count; i+=3){
			Triangle tempTri = Triangle();
			int v0_index = shapes[0].mesh.indices[i  ].vertex_index;
			int v1_index = shapes[0].mesh.indices[i+1].vertex_index;
			int v2_index = shapes[0].mesh.indices[i+2].vertex_index;
			tempTri.v0_index = v0_index;
			tempTri.v1_index = v1_index;
			tempTri.v2_index = v2_index;
			tempTri.morton_code = 0;
			ptr_host_triangles[i/3] = tempTri;
		}
	}

  ~ObjLoader(){
    free(ptr_host_triangles);
  }

  Triangle* ObjLoader::createDeviceTriangleBuffer(){
    Triangle *ptr_device_triangles = nullptr;
    cudaMalloc(&ptr_device_triangles, triangle_count * sizeof(Triangle));
    cudaMemcpy(ptr_device_triangles, ptr_host_triangles, triangle_count * sizeof(Triangle), cudaMemcpyHostToDevice);
    return ptr_device_triangles;
  }

  Vec3* ObjLoader::createDeviceVertexBuffer(){
    Vec3 *ptr_device_vertices = nullptr;
    cudaMalloc(&ptr_device_vertices, vertex_count * sizeof(Vec3));
    cudaMemcpy(ptr_device_vertices, attrib.vertices.data(), vertex_count * sizeof(Vec3), cudaMemcpyHostToDevice);
    return ptr_device_vertices;
  }

  Vec3* ObjLoader::createDeviceNormalBuffer(){
    Vec3 *ptr_device_normals = nullptr;
    cudaMalloc(&ptr_device_normals, normals_count * sizeof(Vec3));
    cudaMemcpy(ptr_device_normals, attrib.normals.data(), normals_count * sizeof(Vec3), cudaMemcpyHostToDevice);
    return ptr_device_normals;
  }

  AABB getSceneBoundingBox(){
	return this->scene_bounding_box;
  }
};