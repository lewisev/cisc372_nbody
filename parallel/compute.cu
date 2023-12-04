#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include "vector.h"
#include "config.h"

//Global values and accelerations
vector3* vals;
vector3** accels;

//Parallel implementation
__global__ void parallelCompute(vector3* vals, vector3** accels, vector3* d_vel, vector3* d_pos, double* d_mass){
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    int i = myId / NUMENTITIES;
    int j = myId % NUMENTITIES;

    accels[myId] = &vals[myId*NUMENTITIES];

    if(myId < NUMENTITIES * NUMENTITIES){
        if(i == j){
            FILL_VECTOR(accels[i][j],0,0,0);
        }else{
            vector3 distance;

            //calculate distance in 3D
            distance[0]=d_pos[i][0]-d_pos[j][0];
            distance[1]=d_pos[i][1]-d_pos[j][1];
            distance[2]=d_pos[i][2]-d_pos[j][2];

            //calculate acceleration values
            //fun fun fun physics calculation stuff
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
            FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }

        vector3 accel_sum = {(double) *(accels[myId])[0], (double) *(accels[myId])[1], (double) *(accels[myId])[2]};

        d_vel[i][0]+=accel_sum[0]*INTERVAL;
		d_pos[i][0]+=d_vel[i][0]*INTERVAL;

		d_vel[i][1]+=accel_sum[1]*INTERVAL;
		d_pos[i][1]+=d_vel[i][1]*INTERVAL;

		d_vel[i][2]+=accel_sum[2]*INTERVAL;
		d_pos[i][2]+=d_vel[i][2]*INTERVAL;
    }
}


//Memory allocation and driver code
void compute(){
    // d_hvel and d_hpos to hold the hVel and hPos variables on the GPU
    vector3 *d_vel, *d_pos;
    double *d_mass;

    // a lil bit slower than managing it ourselves but I cba at this point
    // allocate memory on the device 
    cudaMallocManaged((void**) &d_vel, (sizeof(vector3) * NUMENTITIES));
    cudaMallocManaged((void**) &d_pos, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged((void**) &d_mass, (sizeof(double) * NUMENTITIES));

    // send our information from the device to the gpu
    cudaMemcpy(d_vel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    // allocate space on the gpu for our data
    cudaMallocManaged((void**) &vals, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
    cudaMallocManaged((void**) &accels, sizeof(vector3*)*NUMENTITIES);

    // find the number of blocks that we need to run, then run on the gpu
    int blockSize = 256; 
    int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

    parallelCompute<<<numBlocks, blockSize>>>(vals, accels, d_vel, d_pos, d_mass);
    cudaDeviceSynchronize();

    // copy results from gpu to device
    cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
    cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
    cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDefault);

    // free and we done
    cudaFree(accels);
    cudaFree(vals);
}