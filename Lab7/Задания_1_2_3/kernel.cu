#include <stdio.h>         // Для printf
#include <stdlib.h>        // Для malloc, free, rand
#include <cuda_runtime.h>  // Основные функции CUDA
#include <time.h>          // Для измерения времени CPU

// ЗАДАНИЕ 1: РЕДУКЦИЯ (СУММИРОВАНИЕ)

// Ядро CUDA для редукции (суммирование элементов массива)
__global__ void reductionKernel(float* input, float* output, int n) {
    extern __shared__ float shared[]; // Разделяемая память блока

    int tid = threadIdx.x;                       // Индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid;    // Глобальный индекс элемента массива

    // Копируем элемент из глобальной памяти в shared memory
    if (idx < n) {
        shared[tid] = input[idx];
    }
    else {
        shared[tid] = 0.0f; // Если поток выходит за пределы массива, пишем 0
    }

    __syncthreads(); // Ждем, пока все потоки скопируют данные

    // Древовидная редукция: суммируем элементы попарно
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride]; // Складываем текущий элемент и элемент на расстоянии stride
        }
        __syncthreads(); // Синхронизация перед следующей итерацией
    }

    // Первый поток блока записывает результат в выходной массив
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Функция для редукции на CPU
double cpuReduction(float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)data[i]; // Складываем все элементы массива
    }
    return sum;
}

// Главная функция для редукции на GPU
double gpuReduction(float* h_data, int n) {
    float* d_input = NULL;   // Указатель на данные GPU
    float* d_output = NULL;  // Указатель на выход GPU

    int blockSize = 256;                        // Размер блока потоков
    int gridSize = (n + blockSize - 1) / blockSize; // Количество блоков
    int sharedSize = blockSize * sizeof(float); // Размер shared memory

    // Выделяем память на GPU
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, gridSize * sizeof(float));

    if (d_input == NULL || d_output == NULL) { // Проверка ошибок
        printf("Ошибка: не удалось выделить память на GPU\n");
        return 0;
    }

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_input, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Запускаем ядро редукции
    reductionKernel << <gridSize, blockSize, sharedSize >> > (d_input, d_output, n);

    // Проверка ошибок выполнения ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Ошибка CUDA ядра: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // Ждем завершения всех потоков

    // Копируем результаты блоков с GPU на CPU
    float* h_output = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Складываем результаты всех блоков
    double result = 0.0;
    for (int i = 0; i < gridSize; i++) {
        result += (double)h_output[i];
    }

    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return result; // Возвращаем сумму
}

// ЗАДАНИЕ 2: СКАНИРОВАНИЕ (ПРЕФИКСНАЯ СУММА)

// Ядро CUDA для сканирования (inclusive scan)
__global__ void scanKernel(float* input, float* output, int n) {
    extern __shared__ float scanShared[]; // Shared memory блока

    int tid = threadIdx.x;                   // Индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + tid; // Глобальный индекс

    // Копируем данные в shared memory
    if (idx < n) {
        scanShared[tid] = input[idx];
    }
    else {
        scanShared[tid] = 0.0f;
    }

    __syncthreads(); // Ждем все потоки

    // Алгоритм сканирования (сумма элементов до текущего)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = scanShared[tid - stride]; // Берем элемент на расстоянии stride
        }
        __syncthreads();            // Синхронизация
        if (tid >= stride) {
            scanShared[tid] += temp; // Складываем с текущим
        }
        __syncthreads();            // Синхронизация перед следующим stride
    }

    // Записываем результат в глобальную память
    if (idx < n) {
        output[idx] = scanShared[tid];
    }
}

// Функция CPU для сканирования
void cpuScan(float* input, float* output, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)input[i];   // Накопление суммы
        output[i] = (float)sum;    // Записываем в выходной массив
    }
}

// Ядро для добавления смещения блоков
__global__ void addOffsetKernel(float* output, float* blockSums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс

    // Если блок не первый, добавляем смещение
    if (idx < n && blockIdx.x > 0) {
        output[idx] += blockSums[blockIdx.x - 1];
    }
}

// Главная функция GPU-сканирования
void gpuScan(float* h_input, float* h_output, int n) {
    float* d_input = NULL;
    float* d_output = NULL;
    float* d_blockSums = NULL;

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int sharedSize = blockSize * sizeof(float);

    // Выделяем память на GPU
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));
    cudaMalloc((void**)&d_blockSums, gridSize * sizeof(float));

    // Проверяем память
    if (d_input == NULL || d_output == NULL || d_blockSums == NULL) {
        printf("Ошибка: не удалось выделить память на GPU\n");
        return;
    }

    // Копируем данные на GPU
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // 1) Сканиуем каждый блок
    scanKernel << <gridSize, blockSize, sharedSize >> > (d_input, d_output, n);

    // 2) Считаем суммы последних элементов блоков
    float* h_blockSums = (float*)malloc(gridSize * sizeof(float));
    for (int i = 0; i < gridSize; i++) {
        int lastIdx = ((i + 1) * blockSize - 1 < n) ? ((i + 1) * blockSize - 1) : (n - 1);
        cudaMemcpy(&h_blockSums[i], &d_output[lastIdx], sizeof(float), cudaMemcpyDeviceToHost);
    }

    // 3) Сумма блоков на CPU
    for (int i = 1; i < gridSize; i++) {
        h_blockSums[i] += h_blockSums[i - 1];
    }

    // 4) Копируем суммы блоков на GPU
    cudaMemcpy(d_blockSums, h_blockSums, gridSize * sizeof(float), cudaMemcpyHostToDevice);

    // 5) Добавляем смещение каждому блоку
    addOffsetKernel << <gridSize, blockSize >> > (d_output, d_blockSums, n);

    cudaDeviceSynchronize(); // Ждем GPU

    // Копируем результат на CPU
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    free(h_blockSums);
}

// ГЛАВНАЯ ПРОГРАММА
int main() {
    int sizes[] = { 1024, 2048, 4096, 1000000, 10000000 }; // Размеры тестовых массивов
    int numTests = 5;

    for (int test = 0; test < numTests; test++) {
        int n = sizes[test];
        printf("Размер массива: %d\n", n);

        float* h_data = (float*)malloc(n * sizeof(float));
        float* h_scanOutput = (float*)malloc(n * sizeof(float));
        float* h_cpuScanOutput = (float*)malloc(n * sizeof(float));

        srand(42 + test); // Для воспроизводимости
        for (int i = 0; i < n; i++) {
            h_data[i] = (float)((rand() % 10) + 1); // Случайные числа от 1 до 10
        }

        // ===== РЕДУКЦИЯ =====
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        double gpuResult = gpuReduction(h_data, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime = 0;
        cudaEventElapsedTime(&gpuTime, start, stop);

        clock_t cpuStart = clock();
        double cpuResult = cpuReduction(h_data, n);
        clock_t cpuEnd = clock();
        double cpuTime = ((double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC) * 1000;

        printf("GPU результат: %.0f\n", gpuResult);
        printf("CPU результат: %.0f\n", cpuResult);
        printf("Совпадение: %s\n", (fabs(gpuResult - cpuResult) < cpuResult * 0.001f) ? "ДА" : "НЕТ");
        printf("Время GPU: %.4f мс, CPU: %.4f мс\n", gpuTime, cpuTime);

        // ===== СКАНИРОВАНИЕ =====
        cudaEventRecord(start);
        gpuScan(h_data, h_scanOutput, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&gpuTime, start, stop);
        cpuScan(h_data, h_cpuScanOutput, n);

        bool scanCorrect = true;
        for (int i = 0; i < n; i++) {
            if (fabs(h_scanOutput[i] - h_cpuScanOutput[i]) > h_cpuScanOutput[i] * 0.001f + 0.01f) {
                scanCorrect = false;
                break;
            }
        }

        printf("Сканирование совпадает: %s\n", scanCorrect ? "ДА" : "НЕТ");
        printf("Время GPU: %.4f мс, CPU: %.4f мс\n", gpuTime, cpuTime);

        free(h_data);
        free(h_scanOutput);
        free(h_cpuScanOutput);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("\n---------------------------------------------\n");
    }

    return 0;
}

