#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <assert.h>

template <typename T>
struct BinaryAssociativeOperator {
	__host__ __device__
	virtual T operator() (const T left, const T right) const = 0;
};

template <typename T>
struct Add : public BinaryAssociativeOperator<T>
{
	__host__ __device__
	T operator() (const T left, const T right) const;
};

template <typename T>
struct Max : public BinaryAssociativeOperator<T>
{
	__host__ __device__
	T operator () (const T left, const T right) const;
};

template<typename T, typename Op>
__global__ 
void reduce(T * v, const int n , Op op);

int main(int argc, char** argv) {

	const int size = 8;

	int h_v[size] = { 1, 2, 3, 4, 5, 6 , 7 , 8 };

	int *d_v = 0;

	cudaMalloc((void**)&d_v, size * sizeof(int));

	cudaMemcpy(d_v, h_v, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grdDim(1, 1, 1);
	dim3 blkDim(size / 2, 1, 1);

	// Max
	Max<int> maxOp;
	reduce<int, Max<int>> <<<grdDim, blkDim >> >(d_v, size, maxOp);
	
	cudaDeviceSynchronize();
	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// system("pause"); // when using VisStudio
		exit(-1);
	}

	int max;
	cudaMemcpy(&max, d_v, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_v);

	std::cout << "Max ( ";
	for (int i = 0; i < size; i++) {
		std::cout << h_v[i] << (i < size - 1 ? " ," : " ) = ");
	}
	std::cout << max << std::endl;

	// system("pause"); // when using VisStudio

	return 0;
}

template<typename T>
__host__ __device__
T Add<T>::operator() (const T left, const T right) const {
	return left + right;
};

template<typename T>
__host__ __device__
T Max<T>::operator() (const T left, const T right) const {
	return left > right ? left : right;
};


template<typename T, typename Op>
__global__ 
void reduce(T *v, const int n, Op op)
{

	// Thread Id
	const unsigned int tId = threadIdx.x;
	// Initial thread Count
	const unsigned int initialThreadCount = blockDim.x;

	// assert(initialThreadCount >= n);

	for (int threadCount = initialThreadCount, stepSize = 1;
		threadCount > 0;
		threadCount /= 2, stepSize *= 2) {

		if (tId < threadCount) {
			int indiceDroite = tId * stepSize * 2;
			int indiceGauche = indiceDroite + stepSize;
			v[indiceDroite] = op(v[indiceDroite], v[indiceGauche]);
		}
		__syncthreads();
	}
};
