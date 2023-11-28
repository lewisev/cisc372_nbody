#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//todo: possibly move row and stride to compute function so not defined in each device function??


__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x; //how many operations to do each group
	int i, j, k;

	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=row; i < NUMENTITIES; i += stride) {
		for (j=0; j < NUMENTITIES; j++) {
			if (i == j) {
				FILL_VECTOR(accels[i][j], 0, 0, 0);
			} else {
				vector3 distance;
				for (k=0;k<3;k++) {
					distance[k] = hPos[i][k] - hPos[j][k];
				}
				double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
				double magnitude = sqrt(magnitude_sq);
				double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
				FILL_VECTOR(accels[i][j], accelmag*distance[0]/magnitude, accelmag*distance[1]/magnitude, accelmag*distance[2]/magnitude);
			}
		}
	}
}
__global__ void compute_velocities(vector3 **accels, vector3 *hVel, vector3 *hPos) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x; //how many operations to do each group
	int i, j, k;

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=row; i < NUMENTITIES; i+=stride) {
		vector3 accel_sum = {0, 0, 0};
		for (j=0; j < NUMENTITIES; j++) { // kernel
			for (k=0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k]; //accel_sum local, accels is global
			}
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0; k < 3; k++) { //kernel
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	int num_blocks = 256;
	int block_size = (NUMENTITIES - 1) / num_blocks + 1;


	compute_accels<<<num_blocks, block_size>>>(d_accels, d_hPos, d_mass);
	cudaDeviceSynchronize(); //todo: Maybe not needed if result stays the same without?
	
	compute_velocities<<<num_blocks, block_size>>>(d_accels, d_hVel, d_hPos);
	cudaDeviceSynchronize(); //todo: Maybe not needed if result stays the same without?
	// done in nbody.cu
	/* for (i=0; i < NUMENTITIES; i++) {
		accels[i] =& values[i * NUMENTITIES];
	} */
	
	//free(accels);
	//free(values);
}
