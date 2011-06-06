#include "types.h"

__host__ __device__ float3 operator +(const float3& x, const float3& y)
{
    float3 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;
    return z;
}
__host__ __device__ float3 operator -(const float3& x, const float3& y)
{
    float3 z;
    z.x = x.x - y.x;
    z.y = x.y - y.y;
    z.z = x.z - y.z;
    return z;
}
__host__ __device__ float3 operator *(const float3& x, const float& c)
{
    float3 z;
    z.x = c * y.x;
    z.y = c * y.y;
    z.z = c * y.z;
    return z;
}
__host__ __device__ float3 operator *(const float& c, const float3& x)
{
    return x * c;
}

