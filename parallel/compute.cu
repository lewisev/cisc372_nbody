#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//todo: possibly move Idx/Dim vars to compute function so not defined in each device function??


__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	//int i, j, k;

	//first compute the pairwise accelerations.  Effect is on the first argument.

	if (i == j) {
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	} else {
		__shared__ vector3 distance;
		distance[k] = hPos[i][k] - hPos[j][k];

		__syncthreads(); // sync to make sure all distances have been calculated

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag*distance[0]/magnitude, accelmag*distance[1]/magnitude, accelmag*distance[2]/magnitude);
	}
}

__global__ void compute_velocities(vector3 **accels, vector3 *hVel, vector3 *hPos) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.x * gridDim.x; //how many operations to do each group
	int k = threadIdx.z;

	if(i >= NUMENTITIES) return;

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	/*
	vector3 accel_sum = {0, 0, 0};
	accel_sum[k] += accels[i][j][k]; //accel_sum local, accels is global
	*/


	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	hVel[i][k] += accels[i][j][k] * INTERVAL;
	hPos[i][k] += hVel[i][k] * INTERVAL;
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {

	//todo: remove for loops, calculate sizes for each thing
	
	/* Probably can get rid of these
	int num_blocks = 256;
	int block_size = (NUMENTITIES - 1) / num_blocks + 1; */

	int block_size = (NUMENTITIES - 1) / BLOCK_SIZE + 1;
					//NUMENTITIES -1) / 16 + 1

	dim3 block_dim (BLOCK_SIZE, BLOCK_SIZE, 3);
	dim3 block (block_size, block_size);



	compute_accels<<<block, block_dim>>>(d_accels, d_hPos, d_mass);
	cudaDeviceSynchronize(); //todo: Maybe not needed if result stays the same without?
	

	vector3 **accels;
	cudaMemcpy(accels, d_accels, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	int i,j,k;

	for (i=0; i < NUMENTITIES; i++) {
		vector3 accel_sum = {0, 0, 0};
		for (j=0; j < NUMENTITIES; j++) {
			for (k=0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k];
			}
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}




	//compute_velocities<<<block, block_dim>>>(accels, d_hVel, d_hPos);
	//cudaDeviceSynchronize(); //todo: Maybe not needed if result stays the same without?
	// done in nbody.cu
	/* for (i=0; i < NUMENTITIES; i++) {
		accels[i] =& values[i * NUMENTITIES];
	} */
	
	//free(accels);
	//free(values);
}
