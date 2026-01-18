#include <cuda_runtime.h>
#include <iostream>

#define ARRAY_SIZE 1024       // Размер массива (небольшой для вывода printf)
#define PRINT_LIMIT 16        // Количество элементов для печати

// Kernel для умножения с возможностью печати
__global__ void multiply_kernel(float* arr, float factor, int n, bool print) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] *= factor;

        // Печать первых PRINT_LIMIT элементов
        if (print && idx < PRINT_LIMIT) {
            printf("Thread %d: arr[%d] = %.2f\n", idx, idx, arr[idx]);
        }
    }
}

// Замер времени kernel с помощью CUDA Events
float measure_time(float* d_arr, int n, int threadsPerBlock, float factor, bool print) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Разогрев GPU
    multiply_kernel << <blocks, threadsPerBlock >> > (d_arr, factor, n, false);
    cudaDeviceSynchronize();

    // Замер времени
    cudaEventRecord(start);
    multiply_kernel << <blocks, threadsPerBlock >> > (d_arr, factor, n, print);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    size_t size = ARRAY_SIZE * sizeof(float);
    float* h_arr = new float[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) h_arr[i] = 1.0f * i;

    float* d_arr;
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Массив тестируемых размеров блока
    int threads[] = { 64, 128, 256 };
    int num_configs = sizeof(threads) / sizeof(int);

    float best_time = 1e9;   // Инициализация минимального времени
    float worst_time = 0;    // Инициализация максимального времени
    int best_tpb = 0;        // Оптимальный размер блока
    int worst_tpb = 0;       // Неоптимальный размер блока

    for (int i = 0; i < num_configs; i++) {
        int tpb = threads[i];
        cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);  // Сброс массива

        printf("\n=== Testing threadsPerBlock = %d ===\n", tpb);
        float time = measure_time(d_arr, ARRAY_SIZE, tpb, 2.0f, true);

        printf("Kernel execution time: %.4f ms\n", time);

        // Определяем оптимальный и неоптимальный
        if (time < best_time) {
            best_time = time;
            best_tpb = tpb;
        }
        if (time > worst_time) {
            worst_time = time;
            worst_tpb = tpb;
        }
    }

    printf("\n===============================================\n");
    printf("Summary:\n");
    printf("Optimal configuration: threadsPerBlock = %d, time = %.4f ms\n", best_tpb, best_time);
    printf("Non-optimal configuration: threadsPerBlock = %d, time = %.4f ms\n", worst_tpb, worst_time);
    printf("Speedup: %.2f x\n", worst_time / best_time);
    printf("===============================================\n");

    cudaFree(d_arr);
    delete[] h_arr;

    return 0;
}