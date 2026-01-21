#include <stdio.h>          // стандартная библиотека для ввода/вывода
#include <stdlib.h>         // стандартная библиотека для функций malloc, rand и free
#include <cuda_runtime.h>   // библиотека CUDA Runtime API для работы с GPU
#include <time.h>           // библиотека для работы с time (seed для rand)
#include <math.h>           // математические функции, например fminf, fmaxf

// ===== Ядро REDUCTION: сумма =========
__global__ void reductionSumKernel(float* input, float* output, int n) {
    // extern __shared__ - выделяет разделяемую память на блок
    extern __shared__ float sdata[];
    int tid = threadIdx.x; // локальный индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + tid; // глобальный индекс потока в сетке

    // загружаем данные из глобальной памяти в shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads(); // синхронизируем потоки блока перед редукцией

    // редукция внутри блока (суммирование)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads(); // синхронизируем потоки после каждой итерации
    }

    // первый поток блока записывает результат редукции блока в глобальную память
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ===== Ядро REDUCTION: MIN/MAX =======
__global__ void reductionMinKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // инициализация разделяемой памяти значениями из глобальной памяти или INF для padding
    sdata[tid] = (idx < n) ? input[idx] : INFINITY;
    __syncthreads();

    // редукция MIN внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0]; // записываем результат блока
}

__global__ void reductionMaxKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // инициализация shared memory для MAX
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ===== BLELLOCH SCAN (EXCLUSIVE) ======
__global__ void blellochScanKernel(float* data, float* blockSums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // загружаем данные в shared memory для быстрого доступа
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    int offset = 1;

    // UPSWEEP PHASE: строим "дерево" для редукции
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai]; // суммируем пары элементов
        }
        offset <<= 1;
        __syncthreads();
    }

    // сохраняем сумму блока для мульти-блочного скана и обнуляем последний элемент
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[blockDim.x - 1];
        sdata[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // DOWNSWEEP PHASE: распределяем суммы по элементам блока
    for (int d = 1; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t; // перестановка и добавление для EXCLUSIVE SCAN
        }
        __syncthreads();
    }

    if (idx < n) data[idx] = sdata[tid]; // записываем результат обратно в глобальную память
}

// Добавление cumulative сумм блоков к элементам массива
__global__ void addBlockSumsKernel(float* data, float* blockSums, int n, int blockSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < n) {
        data[idx] += blockSums[blockIdx.x - 1]; // прибавляем сумму всех предыдущих блоков
    }
}

// ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======
// CPU редукция суммы для проверки
double cpuReductionSum(float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data[i];
    return sum;
}

// CPU редукция MIN
float cpuReductionMin(float* data, int n) {
    float minVal = data[0];
    for (int i = 1; i < n; i++) minVal = fminf(minVal, data[i]);
    return minVal;
}

// CPU редукция MAX
float cpuReductionMax(float* data, int n) {
    float maxVal = data[0];
    for (int i = 1; i < n; i++) maxVal = fmaxf(maxVal, data[i]);
    return maxVal;
}

// CPU exclusive scan для проверки
void cpuScanExclusive(float* input, float* output, int n) {
    output[0] = 0.0f; // первый элемент всегда 0
    float sum = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = sum;
        sum += input[i];
    }
}

// CPU inclusive scan для проверки
void cpuScanInclusive(float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// ====== ГЛАВНАЯ ПРОГРАММА ============
int main() {
    int sizes[] = { 1024, 1000000 }; // размеры массивов для тестирования
    int numSizes = 2;

    int blockSizes[] = { 64, 128, 256, 512 }; // размеры блоков для GPU
    int numBlocks = 4;

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        printf("\n==== Размер массива: %d ====\n", n);

        float* h_data = (float*)malloc(n * sizeof(float)); // исходный массив на CPU
        float* h_scanCPU = (float*)malloc(n * sizeof(float)); // результат CPU скана

        srand(42 + s); // фиксируем seed для повторяемости
        for (int i = 0; i < n; i++) h_data[i] = (float)(rand() % 10 + 1); // заполняем числами 1..10

        cpuScanExclusive(h_data, h_scanCPU, n); // CPU scan для проверки

        for (int b = 0; b < numBlocks; b++) {
            int blockSize = blockSizes[b];
            int gridSize = (n + blockSize - 1) / blockSize; // количество блоков на GPU

            float* d_input, * d_sum, * d_min, * d_max;

            cudaMalloc(&d_input, n * sizeof(float));
            cudaMalloc(&d_sum, gridSize * sizeof(float));
            cudaMalloc(&d_min, gridSize * sizeof(float));
            cudaMalloc(&d_max, gridSize * sizeof(float));

            cudaMemcpy(d_input, h_data, n * sizeof(float), cudaMemcpyHostToDevice); // копируем данные на GPU

            cudaEvent_t start, stop; // события для измерения времени GPU
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            reductionSumKernel << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_sum, n); // вызов ядра REDUCTION SUM
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float gpuTime;
            cudaEventElapsedTime(&gpuTime, start, stop); // измеряем время выполнения GPU

            // редукция MIN/MAX
            reductionMinKernel << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_min, n);
            reductionMaxKernel << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_max, n);

            float* h_sum = (float*)malloc(gridSize * sizeof(float));
            float* h_min = (float*)malloc(gridSize * sizeof(float));
            float* h_max = (float*)malloc(gridSize * sizeof(float));

            cudaMemcpy(h_sum, d_sum, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_min, d_min, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_max, d_max, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

            double sumGPU = 0;
            for (int i = 0; i < gridSize; i++) sumGPU += h_sum[i];

            float minGPU = h_min[0];
            float maxGPU = h_max[0];
            for (int i = 1; i < gridSize; i++) {
                minGPU = fminf(minGPU, h_min[i]);
                maxGPU = fmaxf(maxGPU, h_max[i]);
            }

            // выводим результаты
            printf("Block %d: Сумма GPU %.0f CPU %.0f, время %.4f мс\n", blockSize, sumGPU, cpuReductionSum(h_data, n), gpuTime);
            printf("Block %d: MIN GPU %.0f CPU %.0f\n", blockSize, minGPU, cpuReductionMin(h_data, n));
            printf("Block %d: MAX GPU %.0f CPU %.0f\n", blockSize, maxGPU, cpuReductionMax(h_data, n));

            cudaFree(d_input); cudaFree(d_sum); cudaFree(d_min); cudaFree(d_max);
            free(h_sum); free(h_min); free(h_max);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // ==== MULTIBLOCK BLELLOCH SCAN ====
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        float* d_scan, * d_blockSums;
        cudaMalloc(&d_scan, n * sizeof(float));
        cudaMalloc(&d_blockSums, gridSize * sizeof(float));
        cudaMemcpy(d_scan, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

        blellochScanKernel << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_scan, d_blockSums, n); // ядро Blelloch scan
        cudaDeviceSynchronize();

        float* h_blockSums = (float*)malloc(gridSize * sizeof(float));
        cudaMemcpy(h_blockSums, d_blockSums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 1; i < gridSize; i++) h_blockSums[i] += h_blockSums[i - 1]; // cumulative sum по блокам

        cudaMemcpy(d_blockSums, h_blockSums, gridSize * sizeof(float), cudaMemcpyHostToDevice);

        addBlockSumsKernel << <gridSize, blockSize >> > (d_scan, d_blockSums, n, blockSize); // добавляем суммы предыдущих блоков
        cudaDeviceSynchronize();

        float* h_scanGPU = (float*)malloc(n * sizeof(float));
        cudaMemcpy(h_scanGPU, d_scan, n * sizeof(float), cudaMemcpyDeviceToHost);

        int correct = 1;
        float maxError = 0.0f;
        for (int i = 0; i < n; i++) {
            float error = fabs(h_scanGPU[i] - h_scanCPU[i]);
            if (error > maxError) maxError = error;
            if (error > 1e-3) {
                correct = 0;
                if (i < 10) printf("  i=%d: GPU=%.1f CPU=%.1f diff=%.4f\n", i, h_scanGPU[i], h_scanCPU[i], error);
            }
        }
        printf("Blelloch Scan совпадает с CPU: %s (макс ошибка: %.6f)\n", correct ? "ДА" : "НЕТ", maxError);

        free(h_data); free(h_scanCPU); free(h_scanGPU); free(h_blockSums);
        cudaFree(d_scan); cudaFree(d_blockSums);
    }

    return 0;
}
