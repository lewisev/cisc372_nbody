#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

/**
__global__ void fill_accels(vector3 *values, vector3 **accels){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

	//printf("i=%d\n", i);
	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
		//printf("fill accels: %d, %p\n", i,(void*) &values[i * NUMENTITIES]);
	}
}
*/

__global__ void compute_accels(vector3 **accels, vector3 *hPos, double *mass){
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;

	int local_row = threadIdx.y;
	int local_col = threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + local_row;
	int col = (blockIdx.x * blockDim.x) + local_col;


	if(row >= NUMENTITIES || col >= NUMENTITIES) {
		return;
	}


	//printf("row: %d, col: %d\n", i, j);
	//printf("accels[%d]: %p\n", i, (void*) accels[i]);

	if (row == 0 && col == 0) {
		//printf("\n");
	}
	
	if (row == col) {
		FILL_VECTOR(accels[row][col], 0, 0, 0);
		//printf("row: %d col: %d value: %e, %e, %e\n", row, col, accels[row][col][0], accels[row][col][1], accels[row][col][2]);
	} else {
		vector3 distance;
		for (int k = 0; k < 3; k++) {
			distance[k] = hPos[row][k] - hPos[col][k];
		}

		//printf("row %d col %d distance %e %e %e\n", row, col, distance[0], distance[1], distance[2]);

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[col] / magnitude_sq;
		//printf("row %d col %d magnitude %f\n", row, col, magnitude);

		//! good until here - magnitude & accelmag are very close
		
		
		FILL_VECTOR(
			accels[row][col], 
			accelmag * distance[0] / magnitude, 
			accelmag * distance[1] / magnitude, 
			accelmag * distance[2] / magnitude
		);
		

		/*
		accels[row][col][0] = accelmag * distance[0] / magnitude;
		accels[row][col][1] = accelmag * distance[1] / magnitude;
		accels[row][col][2] = accelmag * distance[2] / magnitude;
		*/
		
		/*
		printf("row: %d col: %d value: %f, %f, %f\n", row, col, accelmag*distance[0]/magnitude,
		 accelmag*distance[1]/magnitude, 
		 accelmag*distance[2]/magnitude);
		 */
		//printf("row: %d col: %d value: %f, %f, %f\n", row, col, accels[row][col][0], accels[row][col][1], accels[row][col][2]);
	}
}

__global__ void compute_velocities(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int row = blockIdx.x; // NUMENTITIES
	//int k = threadIdx.x;

	int j, k;

	//printf("row: %d\n", row);

	if(row >= NUMENTITIES) {
		return;
	}

	vector3 accel_sum = {0, 0, 0};
	for (j=0; j < NUMENTITIES; j++) {
		//printf("row: %d\tcol: %d\n", row, j);
		for (k=0; k < 3; k++) {
			accel_sum[k] += accels[row][j][k];
		}
	}
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (k=0; k < 3; k++) {
		hVel[row][k] += accel_sum[k] * INTERVAL;
		hPos[row][k] += hVel[row][k] * INTERVAL;
	}

	/*

	printf("row: %d\n", row);

	if (row == 0) {
		for (int j = 0; j < NUMENTITIES; j++) {
			//printf("row %d col %d {%e %e %e}\n", i, j, accels[i][j][0], accels[i][j][1], accels[i][j][2]);
		}
	}
	*/

	/**
	if(i >= NUMENTITIES) {
		return;
	}
	*/
	/*

	vector3 accel_sum = {0, 0, 0};
	for (int col=0; col < NUMENTITIES; col++){
		for (int k = 0; k < 3; k++) {
			accel_sum[k] += accels[row][col][k];
		}
	}

	for (int k = 0; k < 3; k++) {
		hVel[row][k] += accel_sum[k] * INTERVAL;
		hPos[row][k] += hVel[row][k] * INTERVAL;
	}
	*/
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	dim3 block_size(16,16);
	dim3 block_count((NUMENTITIES+15) / block_size.x, (NUMENTITIES+15) / block_size.y);

	//printf("gridDim: %d %d %d blockDim: %d %d %d\n", block_count.x, block_count.y, block_count.z, block_size.x, block_size.y, block_size.z);
	
	//fill_accels<<<block_count, block_size>>>(values, accels);
	//cudaDeviceSynchronize();

	compute_accels<<<block_count, block_size>>>(accels, d_hPos, d_mass);
	//cudaDeviceSynchronize();

	dim3 grid_dims (NUMENTITIES, 1, 1);

	compute_velocities<<<grid_dims, 1>>>(accels, d_hPos, d_hVel);
	//cudaDeviceSynchronize();
}