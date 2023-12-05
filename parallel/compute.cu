#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

__global__ void fill_accels(vector3 *values, vector3 **accels){
	int i = threadIdx.x;

	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
		printf("fill accels: %d\n, %d", i, &values[i * NUMENTITIES]);
	}
}

__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("i: %d, j: %d\n", i, j);

	//if(i > NUMENTITIES && j > NUMENTITIES) {
	//	return;
	//	printf("return early")
	//}

	if (i == j) {
		printf("try fill (i==j): i: %d, j: %d | accels[%d] = %d\n", i, j, i, accels[i]);
		FILL_VECTOR(accels[i][j], 0, 0, 0);
		printf("fill vector (i==j): i: %d, j: %d\n", i, j);
	} else {
		printf("else - ");
		vector3 distance;
		for (int k = 0; k < 3; k++) {
			distance[k] = hPos[i][k] - hPos[j][k];
		}

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		//printf("before fill in else\n");
		FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		printf("fill vector (else): i: %d, j: %d, magnitude_sq: %d\n", i, j, magnitude_sq);
	}
}

__global__ void compute_velocities(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int i = blockIdx.x;
	int k = threadIdx.x;

	double accel_sum = 0;
	for (int j=0; j < NUMENTITIES; j++){
		accel_sum += accels[i][j][k];
	}

	hVel[i][k] += accel_sum * INTERVAL;
	hPos[i][k] += hVel[i][k] * INTERVAL;
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	// dim3 block_size (16, 16,);
	//int block_size = 256;
	//int block_count = (NUMENTITIES - 1) / block_size + 1;

	dim3 block_size(16,16);
	dim3 block_count((NUMENTITIES+15) / block_size.x, (NUMENTITIES+15) / block_size.y);
	

	fill_accels<<<block_count, block_size>>>(values, accels);
	cudaDeviceSynchronize();

	compute_accels<<<block_count, block_size>>>(accels, d_hPos, d_mass);
	cudaDeviceSynchronize();

	compute_velocities<<<NUMENTITIES, 3>>>(accels, d_hPos, d_hVel);
	cudaDeviceSynchronize();
}