#pragma once
//https://stackoverflow.com/questions/55318151/different-sizes-for-a-struct-in-cpp-and-cuda
#ifdef __CUDACC__
# define ALIGN(x) __align__(x)
#else
# define ALIGN(x) alignas(x)
#endif