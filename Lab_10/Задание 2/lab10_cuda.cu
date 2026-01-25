#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024 * 1024 * 32  // 32 миллиона элементов
#define BLOCK_SIZE 256      // Размер блока потоков

// ==========================
// ЯДРО 1: Коалесцированное чтение
// ==========================
__global__ void gpu_coalesced(float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисление глобального индекса потока
    if (idx < n) {
        // каждый поток обрабатывает свой элемент массива подряд => коалесцированный доступ
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// ==========================
// ЯДРО 2: Некоалесцированное чтение
// ==========================
__global__ void gpu_noncoalesced(float* d_in, float* d_out, int n) {
    int idx = threadIdx.x * gridDim.x + blockIdx.x; // каждый поток прыгает по массиву => некоалесцированный доступ
    if (idx < n) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// ==========================
// ЯДРО 3: Использование shared memory
// ==========================
__global__ void gpu_shared_memory(float* d_in, float* d_out, int n) {
    __shared__ float temp[BLOCK_SIZE]; // локальная shared память для блока

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Загрузка в shared memory
    if (idx < n) {
        temp[threadIdx.x] = d_in[idx];
    }
    __syncthreads(); // синхронизация потоков блока

    // Вычисления с использованием shared memory
    if (idx < n) {
        d_out[idx] = temp[threadIdx.x] * 2.0f;
    }
}

// ==========================
// Функция для замеров времени
// ==========================
void run_kernel(float* d_in, float* d_out, int n, dim3 grid, dim3 block,
    void (*kernel)(float*, float*, int), const char* name) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запуск ядра
    cudaEventRecord(start);
    kernel << <grid, block >> > (d_in, d_out, n);
    cudaEventRecord(stop);

    // Ожидание окончания
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << name << " выполнялось " << milliseconds << " мс" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // ==========================
    // Выделяем память на хосте
    // ==========================
    float* h_in = new float[N];
    float* h_out = new float[N];

    // Инициализация данных
    for (int i = 0; i < N; i++) h_in[i] = i % 100;

    // ==========================
    // Выделяем память на GPU
    // ==========================
    float* d_in, * d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // ==========================
    // Настройка сетки и блоков
    // ==========================
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ==========================
    // Запуск ядер
    // ==========================
    run_kernel(d_in, d_out, N, grid, block, gpu_coalesced, "Коалесцированное ядро");
    run_kernel(d_in, d_out, N, grid, block, gpu_noncoalesced, "Некоалесцированное ядро");
    run_kernel(d_in, d_out, N, grid, block, gpu_shared_memory, "Ядро с shared memory");

    // ==========================
    // Копирование результата обратно
    // ==========================
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ==========================
    // Проверка правильности
    // ==========================
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i] * 2.0f) {
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Результат проверки: корректно!" << std::endl;
    else    std::cout << "Ошибка в вычислениях!" << std::endl;

    // ==========================
    // Очистка памяти
    // ==========================
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
