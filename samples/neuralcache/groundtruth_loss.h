#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>

__device__ inline float l2_loss(float prediction, float target)
{
	const float difference = prediction - target;
	return difference * difference;
}

__device__ inline float relative_l2_loss(float prediction, float target)
{
	const float prediction_sq_plus_epsilon = prediction * prediction + 0.01f;
	const float difference = prediction - target;
	return difference * difference / prediction_sq_plus_epsilon;
}
