#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

__global__ void fill_accels(vector3 *values, vector3 **accels){
	int i = threadIdx.x;

	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
	}
}

__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < NUMENTITIES && j < NUMENTITIES) {
		return;
	}

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
	dim3 block_count((NUMENTITIES+15) / blocksize.x, (NUMENTITIES+15) / blocksize.y);

	fill_accels<<<block_count, block_size>>>(values, accels);

	compute_accels<<<block_count, block_size>>>(accels, d_hPos, d_mass);

	compute_velocities<<<block_count, block_size>>>(accels, d_hPos, d_hVel);
}