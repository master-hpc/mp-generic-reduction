#include <iostream>
#include <assert.h>

__global__ void reduce(int* v, const int n);

int main(int argc, char** argv) {

	const int size = 8;

	int h_v[size] = { 1, 2, 3, 4, 5, 6 , 7 , 8};

	int *d_v = 0;

	cudaMalloc((void**)&d_v, size * sizeof(int));

	cudaMemcpy(d_v, h_v, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grdDim(1, 1, 1);
	dim3 blkDim(size/2, 1, 1);	

	reduce <<<grdDim, blkDim>>>(d_v, size);

	int sum;
	cudaMemcpy(&sum, d_v, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "Somme ( ";
	for (int i = 0; i < size; i++) {
		std::cout << h_v[i] << (i < size -1 ? " ," : " ) = ");
	}
	std::cout << sum << std::endl;

	return 0;
}


__global__ void reduce(int *v, const int n)
{
	
	// Thread Id
        const int tId = threadIdx.x;
	// Initial thread Count
	const int initialThreadCount = blockDim.x;
	
	// assert(initialThreadCount >= n);
	
	for (int threadCount = initialThreadCount, stepSize = 1;
		threadCount > 0;
		threadCount /= 2, stepSize *=2) {
		
		if (tId < threadCount) {
			int indiceDroite = tId * stepSize * 2;
			int indiceGauche = indiceDroite + stepSize;
			v[indiceDroite] += v[indiceGauche];
		}
		__syncthreads();
	}
}
