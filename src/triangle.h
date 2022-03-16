struct Triangle{
    public:
    int morton_code;
    int v0_index;
    int v1_index;
    int v2_index;
    // __device__ __host__ Triangle::Triangle(int v0_index, int v1_index, int v2_index, int morton_code){
    //     this->morton_code = morton_code;
    //     ptr_v0_index = v0_index;
    //     ptr_v1_index = v1_index;
    //     ptr_v2_index = v2_index;
    // }
};
__host__ __device__ inline bool operator< (const Triangle& lhs, const Triangle& rhs){ return lhs.morton_code < rhs.morton_code; }
__host__ __device__ inline bool operator> (const Triangle& lhs, const Triangle& rhs){ return rhs < lhs; }
__host__ __device__ inline bool operator<=(const Triangle& lhs, const Triangle& rhs){ return !(lhs > rhs); }
__host__ __device__ inline bool operator>=(const Triangle& lhs, const Triangle& rhs){ return !(lhs < rhs); }