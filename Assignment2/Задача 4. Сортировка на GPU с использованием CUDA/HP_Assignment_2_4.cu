#include <iostream>           // Для ввода/вывода
#include <vector>             // Для std::vector
#include <cstdlib>            // Для rand()
#include <chrono>             // Для измерения времени
#include <climits>            // Для INT_MAX

#include <cuda_runtime.h>     // Основные функции CUDA runtime API

#define BLOCK_SIZE 256        // Размер блока потоков для сортировки внутри блока

// GPU: bitonic sort для одного блока
__global__ void bitonic_block_sort(int* data, int n) {
    __shared__ int s[BLOCK_SIZE];              // Shared memory для ускоренной сортировки
    int tid = threadIdx.x;                     // Локальный индекс потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid;   // Глобальный индекс потока по всему массиву

    // Копируем данные в shared memory, если поток не выходит за границы массива
    if (gid < n) s[tid] = data[gid];
    else s[tid] = INT_MAX;                      // Если поток вне массива, ставим INT_MAX

    __syncthreads();                           // Синхронизация потоков

    // Bitonic sort в shared memory
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;                 // Вычисление индекса для обмена
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s[tid] > s[ixj]) {    // Восходящая сортировка
                        int tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
                else {
                    if (s[tid] < s[ixj]) {    // Нисходящая сортировка
                        int tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
            }
            __syncthreads();                   // Синхронизация после каждой стадии
        }
    }

    // Копируем отсортированный блок обратно в глобальную память
    if (gid < n) data[gid] = s[tid];
}

// GPU: попарное слияние блоков
__global__ void merge_pair_kernel(int* src, int* dst, int n, int runSize) {
    int pairId = blockIdx.x;                     // Номер пары блоков для слияния
    int leftStart = pairId * (runSize * 2);      // Начало левого блока
    int mid = leftStart + runSize;
    int rightStart = mid;
    int leftEnd = min(mid, n);
    int rightEnd = min(rightStart + runSize, n);

    if (threadIdx.x == 0) {                      // Один поток выполняет последовательное слияние
        int i = leftStart;
        int l = leftStart;
        int r = rightStart;
        while (l < leftEnd && r < rightEnd) {
            if (src[l] <= src[r]) dst[i++] = src[l++];
            else dst[i++] = src[r++];
        }
        while (l < leftEnd) dst[i++] = src[l++];
        while (r < rightEnd) dst[i++] = src[r++];
    }
}

// GPU MergeSort wrapper
float gpu_merge_sort(std::vector<int>& a) {
    int n = static_cast<int>(a.size());
    if (n == 0) return 0.0f;

    int* d_src = nullptr;
    int* d_tmp = nullptr;
    size_t bytes = n * sizeof(int);

    cudaMalloc(&d_src, bytes);                     // Выделяем память на GPU
    cudaMalloc(&d_tmp, bytes);                     // Временный буфер
    cudaMemcpy(d_src, a.data(), bytes, cudaMemcpyHostToDevice); // Копируем данные на GPU

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1) Сортировка каждого блока параллельно
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bitonic_block_sort << <blocks, BLOCK_SIZE >> > (d_src, n);
    cudaDeviceSynchronize();

    // 2) Итеративное попарное слияние блоков
    int runSize = BLOCK_SIZE;
    int* d_read = d_src;
    int* d_write = d_tmp;

    while (runSize < n) {
        int pairs = (n + runSize * 2 - 1) / (runSize * 2);
        merge_pair_kernel << <pairs, BLOCK_SIZE >> > (d_read, d_write, n, runSize);
        cudaDeviceSynchronize();
        // Меняем местами буферы
        int* t = d_read; d_read = d_write; d_write = t;
        runSize *= 2;
    }

    if (d_read != d_src) cudaMemcpy(d_src, d_read, bytes, cudaMemcpyDeviceToDevice);

    // Копируем результат на CPU
    cudaMemcpy(a.data(), d_src, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_src); cudaFree(d_tmp);

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t_end - t_start).count(); // Возвращаем время
}

// Бенчмарк
void benchmark(int n) {
    std::vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = rand();

    std::cout << "\nРазмер массива: " << n << "\n";

    float time = gpu_merge_sort(a);                 // Запуск GPU MergeSort
    std::cout << "GPU MergeSort time: " << time << " ms\n";
}

// main
int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);              // Проверка доступных GPU
    if (deviceCount == 0) {
        std::cerr << "CUDA устройство не обнаружено!\n";
        return 1;
    }
    else {
        std::cout << "Detected " << deviceCount << " CUDA device(s)\n";
    }

    benchmark(10000);       // 10 000 элементов
    benchmark(100000);      // 100 000 элементов

    return 0;
}
