#include <sys/timeb.h>
#include <cmath>
#include <cstdio>

#define BLOCK_SIZE 512

__device__ double polynominal(double x){
    return 5*pow(x,4) + 4*pow(x,3) + x - 10*pow(x,2);
}

__global__ void  calculate(double* result, double start, double dx, long long int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;;

    double x = start + (double(index) * dx);

    if (index < (length)) {
        result[index] = (polynominal(x) + polynominal(x + dx)) * dx / 2.0f;
    }
}

__host__ double integral(double start, double stop, double dx) {
    double sum = 0;

    int length = (int)((stop - start) / dx) ;
    printf("Length in: %d\n", length);

    int size = length * sizeof(double);

    double* hostData = (double*)malloc(size);
    double* deviceData;
    cudaMalloc((void**)&deviceData, size);

    int blocksCount = length / BLOCK_SIZE;

    if((length % BLOCK_SIZE) > 0) {
        blocksCount++;
    }

    calculate<<<blocksCount, BLOCK_SIZE>>>(deviceData, start, dx, length);

    cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < length; i++) {
        sum += hostData[i];
    }

    free(hostData);
    cudaFree(deviceData);
    return sum;
}



int main() {
    struct timeb startTIme{}, stopTime{};

    ftime(&startTIme);

    double result = integral(0, 100, 0.001);

    ftime(&stopTime);

    auto time = stopTime.time - startTIme.time;

    printf("Integral: %.3f\n", result);

    printf("Done in: %d ms\n", time);
}