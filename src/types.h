#pragma once

#include "vector_types.h"

typedef uchar4 rgb;

typedef float3 vertex; // x, h, z

typedef float3 gridpoint; // u, h, v

__host__ __device__ float3 operator +(const float3& x, const float3& y);
__host__ __device__ float3 operator -(const float3& x, const float3& y);
__host__ __device__ float3 operator *(const float3& x, const float& c);
__host__ __device__ float3 operator *(const float& c, const float3& x);
