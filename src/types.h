#pragma once

#include "vector_types.h"

typedef uchar4 rgb;

typedef float3 vertex;

typedef float2 velocity;

__device__ float3 operator +(const float3& x, const float3& y);
__device__ float3 operator -(const float3& x, const float3& y);
__device__ float3 operator *(const float3& x, const float& c);
__device__ float3 operator *(const float& c, const float3& x);