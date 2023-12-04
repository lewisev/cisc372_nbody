#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"
//make an acceleration matrix which is NUMENTITIES squared in size;
__global__ void constructAccels(vector3* values, vector3** accels){
        int in = threadIdx.x;
        if(in < NUMENTITIES){
            accels[in]=&values[in*NUMENTITIES];
        }
}

//first compute the pairwise accelerations.  Effect is on the first argument.
__global__ void computePairwiseAccels(vector3 **accels, vector3 *values, vector3 *hPos, vector3 *hvel, double *mass){
	int in = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int i, j, k;

	for (i=in;i<NUMENTITIES;i+=stride){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
}

//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
__global__ void computeSum(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int in = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int i, j, k;

	for (i=in;i<NUMENTITIES;i+=stride){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	int blockSize = 256;
	int numBlocks = (NUMENTITIES - 1)/blockSize+1;
	
	vector3* values;
	vector3** accels;
	double* d_mass;
	
	//allocate memory
	cudaMalloc((void**)&values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
    cudaMalloc((void**)&accels,sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&d_mass,sizeof(double));
	cudaMalloc((void**)&d_hPos,sizeof(vector3)*NUMENTITIES);
    cudaMalloc((void**)&d_hVel,sizeof(vector3)*NUMENTITIES);

	//copy to the device
	cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel,hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,mass,sizeof(double),cudaMemcpyHostToDevice);

	constructAccels<<<numBlocks, blockSize>>>(values, accels);
	cudaDeviceSynchronize();

	computePairwiseAccels<<<numBlocks, blockSize>>>(accels, values, d_hPos, d_hVel, d_mass);
	cudaDeviceSynchronize();

	computeSum<<<numBlocks, blockSize>>>(accels, d_hPos, d_hVel);
	cudaDeviceSynchronize();

	//copy results to host 
	cudaMemcpy(hPos,d_hPos,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel,d_hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
    cudaMemcpy(mass,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(accels);
    cudaFree(values);
    cudaFree(d_mass);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
}