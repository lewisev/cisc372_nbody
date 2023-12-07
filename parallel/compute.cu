#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"


__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass){
	int local_row = threadIdx.y;
	int local_col = threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + local_row;
	int col = (blockIdx.x * blockDim.x) + local_col;

	if(row >= NUMENTITIES || col >= NUMENTITIES) {
		return;
	}

	//first compute the pairwise accelerations.  Effect is on the first argument.
	if (row == col) {
		FILL_VECTOR(accels[row][col], 0, 0, 0);
	} else {
		vector3 distance;
		for (int k = 0; k < 3; k++) {
			distance[k] = hPos[row][k] - hPos[col][k];
		}

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[col] / magnitude_sq;

		FILL_VECTOR(
			accels[row][col], 
			accelmag * distance[0] / magnitude, 
			accelmag * distance[1] / magnitude, 
			accelmag * distance[2] / magnitude
		);
	}
}


__global__ void compute_velocities(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int row = blockIdx.x; // 0 to NUMENTITIES
	int j;
	int k = threadIdx.x; // 0 to 3

	if(row >= NUMENTITIES) {
		return;
	}

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	vector3 accel_sum = {0, 0, 0};
	for (j=0; j < NUMENTITIES; j++) {
		accel_sum[k] += accels[row][j][k];
	}

	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	hVel[row][k] += accel_sum[k] * INTERVAL;
	hPos[row][k] += hVel[row][k] * INTERVAL;
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	dim3 block_size(16,16);
	dim3 block_count((NUMENTITIES+15) / block_size.x, (NUMENTITIES+15) / block_size.y);
	compute_accels<<<block_count, block_size>>>(accels, d_hPos, d_mass);

	dim3 grid_dims (NUMENTITIES, 1, 1);
	compute_velocities<<<grid_dims, 3>>>(accels, d_hPos, d_hVel);
}