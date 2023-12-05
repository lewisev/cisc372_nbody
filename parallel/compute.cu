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
		//printf("fill accels: %d, %p\n", i,(void*) &values[i * NUMENTITIES]);
	}
}

__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;


	if(i >= NUMENTITIES || j >= NUMENTITIES) {
		return;
	}

	//printf("i: %d, j: %d\n", i, j);
	//printf("accels[%d]: %p\n", i, (void*) accels[i]);
	
	if (i == j) {
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	} else {
		vector3 distance;
		for (int k = 0; k < 3; k++) {
			distance[k] = hPos[i][k] - hPos[j][k];
		}

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
}

__global__ void compute_velocities(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.x * gridDim.x; //how many operations to do each group
	int k;

	if(i >= NUMENTITIES || j >= NUMENTITIES) {
		return;
	}

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	vector3 accel_sum = {0, 0, 0};
	for (int k = 0; k < 3; k++) {
		accel_sum[k] += accels[i][j][k]; //accel_sum local, accels is global
	
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		hVel[i][k] += accel_sum[k] * INTERVAL;
		hPos[i][k] += hVel[i][k] * INTERVAL;
	}
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	dim3 block_size(16,16);
	dim3 block_count((NUMENTITIES+15) / block_size.x, (NUMENTITIES+15) / block_size.y);
	

	fill_accels<<<1, NUMENTITIES>>>(values, accels);
	//cudaDeviceSynchronize();

	compute_accels<<<block_count, block_size>>>(accels, d_hPos, d_mass);
	//cudaDeviceSynchronize();

	compute_velocities<<<NUMENTITIES, 3>>>(accels, d_hPos, d_hVel);
	//cudaDeviceSynchronize();
}