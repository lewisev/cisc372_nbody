#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

//todo: possibly move Idx/Dim vars to compute function so not defined in each device function??


__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	//int i, j, k;

	//first compute the pairwise accelerations.  Effect is on the first argument.
	if(i >= NUMENTITIES || j >= NUMENTITIES) {
		return;
	}
	if (i == j) {
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	} else {
		vector3 distance;
		distance[k] = hPos[i][k] - hPos[j][k];

		__syncthreads(); // sync to make sure all distances have been calculated

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag*distance[0]/magnitude, accelmag*distance[1]/magnitude, accelmag*distance[2]/magnitude);
	}
}

__global__ void compute_velocities(vector3 **accels, vector3 *hVel, vector3 *hPos) {
	/* //int i = threadIdx.x + blockIdx.x * blockDim.x;
	//int j = threadIdx.y + blockDim.x * gridDim.x; //how many operations to do each group
	//int k = threadIdx.z;
	int i=blockIdx.x;
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	vector3 accel_sum = {0, 0, 0};
	accel_sum[k] += accels[i][j][k]; //accel_sum local, accels is global
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	hVel[i][k] += accel_sum[k] * INTERVAL;
	hPos[i][k] += hVel[i][k] * INTERVAL; */
	
	for (int i=0; i < NUMENTITIES; i++){
		vector3 accel_sum = {0, 0, 0};
		for (int j=0; j < NUMENTITIES; j++) {
			for (int k=0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k];
			}
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k=0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

//compute: Updates the positions and locations of the objects in the system nbased on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {	
	
	dim3 square_block_dim (BLOCK_SIZE, BLOCK_SIZE, 3);
	dim3 block_dim ((NUMENTITIES+15)/16, (NUMENTITIES+15)/16, 3);

	compute_accels<<<square_block_dim, block_dim>>>(d_accels, d_hPos, d_mass);
	
	compute_velocities<<<NUMENTITIES, 1>>>(d_accels, d_hVel, d_hPos);
	//DO we need these???
	cudaMemcpy(hVel, d_Vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_Pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
}
