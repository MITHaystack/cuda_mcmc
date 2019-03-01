//
// Compile:
//
// $ nvcc list_gpus.cu -o list_gpus
//
//

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device Index %d, %s, Compute Capability %d.%d\n",
               device, deviceProp.name, deviceProp.major, deviceProp.minor);
    }    
}
