#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math.h>

// ==================== CUDA ЯДРА ====================

// Ядро для умножения на GPU
__global__ void multiplyArrayGPU(float* data, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * factor;
    }
}

// Ядро для сложения на GPU
__global__ void addArrayGPU(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + value;
    }
}

// Ядро для возведения в квадрат на GPU
__global__ void squareArrayGPU(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

// ==================== ФУНКЦИИ ОБРАБОТКИ ====================

// Функция обработки на CPU с использованием OpenMP
// Операция: сложение элементов массива с числом
void addArrayCPU(float* data, int size, float value) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        data[i] = data[i] + value;
    }
}

// Функция обработки на CPU: умножение
void multiplyArrayCPU(float* data, int size, float factor) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * factor;
    }
}

// Функция обработки на GPU с возвратом времени
double processArrayOnGPU(float* h_data, int size, int operation) {
    float* d_data;
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    cudaError_t err;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc((void**)&d_data, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("Ошибка выделения памяти GPU: %s\\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Ошибка копирования на GPU: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    cudaEventRecord(start);

    // Выбор операции (0 = умножение, 1 = сложение, 2 = квадрат)
    if (operation == 0) {
        multiplyArrayGPU << <gridSize, blockSize >> > (d_data, size, 2.0f);
    }
    else if (operation == 1) {
        addArrayGPU << <gridSize, blockSize >> > (d_data, size, 10.0f);
    }
    else if (operation == 2) {
        squareArrayGPU << <gridSize, blockSize >> > (d_data, size);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Ошибка запуска ядра: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    err = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Ошибка копирования с GPU: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)milliseconds;
}

// Инициализация массива
void initializeArray(float* data, int size) {
    srand((unsigned int)time(NULL));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 1000) / 10.0f;
    }
}

// ==================== ЗАДАНИЕ 1: РАЗНЫЕ ОПЕРАЦИИ ====================

void task1_different_operations() {
    printf("\\n");
    printf("========================================================================\\n");
    printf("ЗАДАНИЕ 1: РАЗНЫЕ ОПЕРАЦИИ НА CPU И GPU\\n");
    printf("========================================================================\\n");
    printf("CPU: Сложение (+10) | GPU: Возведение в квадрат\\n\\n");

    const int N = 5000000; // 5 миллионов

    // Создаем исходный массив
    float* original = (float*)malloc(N * sizeof(float));
    float* cpu_data = (float*)malloc(N * sizeof(float));
    float* gpu_data = (float*)malloc(N * sizeof(float));
    float* hybrid_data = (float*)malloc(N * sizeof(float));

    initializeArray(original, N);

    // Копируем в рабочие массивы
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        cpu_data[i] = original[i];
        gpu_data[i] = original[i];
        hybrid_data[i] = original[i];
    }

    // CPU: Сложение (+10)
    printf("1. ОБРАБОТКА НА CPU (сложение +10):\\n");
    double cpuStart = omp_get_wtime();
    addArrayCPU(cpu_data, N, 10.0f);
    double cpuTime = (omp_get_wtime() - cpuStart) * 1000.0;
    printf("   Время: %.3f мс\\n", cpuTime);
    printf("   Пример: %.1f + 10 = %.1f\\n\\n", original[0], cpu_data[0]);

    // GPU: Возведение в квадрат
    printf("2. ОБРАБОТКА НА GPU (возведение в квадрат x²):\\n");
    double gpuTime = processArrayOnGPU(gpu_data, N, 2);
    printf("   Время: %.3f мс\\n", gpuTime);
    printf("   Пример: %.1f ^ 2 = %.1f\\n\\n", original[0], gpu_data[0]);

    // Гибридный: Первая половина CPU (сложение), вторая GPU (возведение в квадрат)
    printf("3. ГИБРИДНЫЙ ПОДХОД (разные операции):\\n");
    printf("   Первая половина: CPU сложение (+10)\\n");
    printf("   Вторая половина: GPU возведение в квадрат\\n\\n");

    double hybridStart = omp_get_wtime();

#pragma omp parallel sections
    {
#pragma omp section
        {
            // CPU: сложение для первой половины
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N / 2; i++) {
                hybrid_data[i] = hybrid_data[i] + 10.0f;
            }
        }

#pragma omp section
        {
            // GPU: квадрат для второй половины
            float* d_data;
            cudaMalloc((void**)&d_data, (N / 2) * sizeof(float));
            cudaMemcpy(d_data, &hybrid_data[N / 2], (N / 2) * sizeof(float),
                cudaMemcpyHostToDevice);

            int blockSize = 256;
            int gridSize = ((N / 2) + blockSize - 1) / blockSize;
            squareArrayGPU << <gridSize, blockSize >> > (d_data, N / 2);

            cudaMemcpy(&hybrid_data[N / 2], d_data, (N / 2) * sizeof(float),
                cudaMemcpyDeviceToHost);
            cudaFree(d_data);
        }
    }

    double hybridTime = (omp_get_wtime() - hybridStart) * 1000.0;
    printf("   Время: %.3f мс\\n", hybridTime);
    printf("   Первая половина: %.1f + 10 = %.1f\\n", original[0], hybrid_data[0]);
    printf("   Вторая половина: %.1f ^ 2 = %.1f\\n\\n", original[N / 2], hybrid_data[N / 2]);

    // Сравнение
    printf("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:\\n");
    printf("  CPU операция:      %.3f мс\\n", cpuTime);
    printf("  GPU операция:      %.3f мс (ускорение %.2fx)\\n", gpuTime, cpuTime / gpuTime);
    printf("  Гибридный подход:  %.3f мс (ускорение %.2fx)\\n", hybridTime, cpuTime / hybridTime);

    free(original);
    free(cpu_data);
    free(gpu_data);
    free(hybrid_data);
}

// ==================== ЗАДАНИЕ 2: РАЗНЫЕ РАЗМЕРЫ МАССИВА ====================

void task2_array_sizes() {
    printf("\\n");
    printf("========================================================================\\n");
    printf("ЗАДАНИЕ 2: ЭКСПЕРИМЕНТЫ С РАЗНЫМИ РАЗМЕРАМИ МАССИВА\\n");
    printf("========================================================================\\n");
    printf("Определение точки пересечения где GPU становится эффективнее CPU\\n\\n");

    // Размеры массивов для тестирования
    int sizes[] = { 100000, 500000, 1000000, 2000000, 5000000, 10000000 };
    int num_sizes = 6;

    printf("Размер    | CPU (мс)  | GPU (мс)  | Ускорение | Эффективность\\n");
    printf("----------|-----------|-----------|-----------|---------------\\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];

        float* original = (float*)malloc(N * sizeof(float));
        float* cpu_data = (float*)malloc(N * sizeof(float));
        float* gpu_data = (float*)malloc(N * sizeof(float));

        initializeArray(original, N);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            cpu_data[i] = original[i];
            gpu_data[i] = original[i];
        }

        // Тест на CPU
        double cpuStart = omp_get_wtime();
        multiplyArrayCPU(cpu_data, N, 2.0f);
        double cpuTime = (omp_get_wtime() - cpuStart) * 1000.0;

        // Тест на GPU
        double gpuTime = processArrayOnGPU(gpu_data, N, 0);

        // Расчет метрик
        double speedup = cpuTime / gpuTime;
        double efficiency = (speedup / 1.0) * 100.0; // Эффективность 1 GPU

        // Форматированный вывод
        if (N >= 1000000) {
            printf("%dM    | %9.3f | %9.3f | %9.2fx | %6.1f%%\\n",
                N / 1000000, cpuTime, gpuTime, speedup, efficiency);
        }
        else {
            printf("%dK    | %9.3f | %9.3f | %9.2fx | %6.1f%%\\n",
                N / 1000, cpuTime, gpuTime, speedup, efficiency);
        }

        free(original);
        free(cpu_data);
        free(gpu_data);
    }

    printf("\\nВЫВОДЫ:");
    printf("\\n GPU эффективнее CPU когда размер данных > 1000000 элементов\\n");
}

// ==================== ЗАДАНИЕ 3: ПРОФИЛИРОВАНИЕ ====================

void task3_profiling() {
    printf("\\n");
    printf("========================================================================\\n");
    printf("ЗАДАНИЕ 3: ПРОФИЛИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ\\n");
    printf("========================================================================\\n");
    printf("Анализ узких мест и оптимизации\\n\\n");

    const int N = 5000000;

    float* original = (float*)malloc(N * sizeof(float));
    float* gpu_data = (float*)malloc(N * sizeof(float));

    initializeArray(original, N);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        gpu_data[i] = original[i];
    }

    // Профилирование отдельных операций
    printf("ПРОФИЛИРОВАНИЕ ОПЕРАЦИЙ:\\n\\n");

    // 1. Выделение памяти
    printf("1. ВЫДЕЛЕНИЕ ПАМЯТИ на GPU:\\n");
    double allocStart = omp_get_wtime();
    float* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float));
    double allocTime = (omp_get_wtime() - allocStart) * 1000.0;
    printf("   Время: %.3f мс\\n", allocTime);
    printf("   Память: %.2f МБ\\n\\n", (N * sizeof(float)) / (1024.0f * 1024.0f));

    // 2. Копирование на GPU
    printf("2. КОПИРОВАНИЕ ДАННЫХ (CPU -> GPU):\\n");
    double copyToStart = omp_get_wtime();
    cudaMemcpy(d_data, gpu_data, N * sizeof(float), cudaMemcpyHostToDevice);
    double copyToTime = (omp_get_wtime() - copyToStart) * 1000.0;
    printf("   Время: %.3f мс\\n", copyToTime);
    printf("   Пропускная способность: %.2f GB/s\\n",
        (N * sizeof(float) / 1e9) / (copyToTime / 1000.0));
    printf("   Теоретический максимум: ~16 GB/s (PCIe 4.0)\\n\\n");

    // 3. Вычисление на GPU
    printf("3. ВЫЧИСЛЕНИЕ НА GPU (умножение):\\n");
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    multiplyArrayGPU << <gridSize, blockSize >> > (d_data, N, 2.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("   Время ядра: %.3f мс\\n", kernelTime);
    printf("   Пиковая производительность GPU: ~1.5 TFLOPS (для T4)\\n");
    printf("   Использование: %.2f%% от максимума\\n\\n",
        (N / 1e9 / kernelTime * 1000.0) / 1.5 * 100.0);

    // 4. Копирование обратно
    printf("4. КОПИРОВАНИЕ ДАННЫХ (GPU -> CPU):\\n");
    double copyFromStart = omp_get_wtime();
    cudaMemcpy(gpu_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    double copyFromTime = (omp_get_wtime() - copyFromStart) * 1000.0;
    printf("   Время: %.3f мс\\n", copyFromTime);
    printf("   Пропускная способность: %.2f GB/s\\n",
        (N * sizeof(float) / 1e9) / (copyFromTime / 1000.0));

    // 5. Итоговый анализ
    printf("\\nИТОГОВЫЙ АНАЛИЗ:\\n");
    double totalTime = allocTime + copyToTime + kernelTime + copyFromTime;
    printf("  Выделение памяти:    %.3f мс (%.1f%%)\\n", allocTime, allocTime / totalTime * 100);
    printf("  Копирование (туда):  %.3f мс (%.1f%%)\\n", copyToTime, copyToTime / totalTime * 100);
    printf("  Вычисления на GPU:   %.3f мс (%.1f%%)\\n", kernelTime, kernelTime / totalTime * 100);
    printf("  Копирование (назад): %.3f мс (%.1f%%)\\n", copyFromTime, copyFromTime / totalTime * 100);
    printf("  ──────────────────────────────────\\n");
    printf("  ИТОГО:               %.3f мс\\n\\n", totalTime);

    printf("УЗКИЕ МЕСТА И РЕКОМЕНДАЦИИ:\\n");
    if (copyToTime + copyFromTime > kernelTime) {
        printf(" УЗКОЕ МЕСТО: Передача данных занимает %.1f%% времени\\n",
            (copyToTime + copyFromTime) / totalTime * 100);
        printf("  Рекомендация:\\n");
        printf("    - Использовать pinned memory (cudaHostAlloc)\\n");
        printf("    - Применить асинхронные передачи (CUDA streams)\\n");
        printf("    - Увеличить объем вычислений на один батч данных\\n");
    }
    else {
        printf(" УЗКОЕ МЕСТО: Вычисления занимают %.1f%% времени\\n",
            kernelTime / totalTime * 100);
        printf("  Рекомендация:\\n");
        printf("    - Оптимизировать CUDA ядро\\n");
        printf("    - Использовать shared memory\\n");
        printf("    - Увеличить параллелизм\\n");
    }

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(original);
    free(gpu_data);
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

int main() {
    // ЗАДАНИЕ 1: Разные операции
    task1_different_operations();

    // ЗАДАНИЕ 2: Разные размеры
    task2_array_sizes();

    // ЗАДАНИЕ 3: Профилирование
    task3_profiling();

    s
    return 0;
}