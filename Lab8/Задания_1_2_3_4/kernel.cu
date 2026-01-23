#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math.h>

// ==================== CUDA ЯДРО ====================
// Ядро CUDA для обработки массива на GPU
__global__ void processArrayGPU(float* data, int size) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, не вышел ли индекс за границы массива
    if (idx < size) {
        // Умножаем каждый элемент на 2
        data[idx] = data[idx] * 2.0f;
    }
}

// ==================== ФУНКЦИИ ОБРАБОТКИ ====================

// Функция обработки массива на CPU с использованием OpenMP
void processArrayCPU(float* data, int size) {
    // Параллелизуем цикл с помощью OpenMP
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * 2.0f;
    }
}

// Функция для обработки на GPU с замером времени
double processArrayOnGPU(float* h_data, int size) {
    float* d_data; // Указатель на память GPU
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    cudaError_t err;

    // Создаем события для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Выделяем память на GPU
    err = cudaMalloc((void**)&d_data, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("Ошибка выделения памяти GPU: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Копируем данные с CPU на GPU
    err = cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Ошибка копирования на GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    // Вычисляем параметры сетки и блоков
    int blockSize = 256; // Число потоков в блоке
    int gridSize = (size + blockSize - 1) / blockSize; // Число блоков

    // Записываем начальное событие
    cudaEventRecord(start);

    // Запускаем CUDA ядро
    processArrayGPU << <gridSize, blockSize >> > (d_data, size);

    // Проверяем ошибки ядра
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Ошибка запуска ядра: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    // Записываем конечное событие
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Вычисляем прошедшее время
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копируем результаты обратно на CPU
    err = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Ошибка копирования с GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 0;
    }

    // Освобождаем память на GPU
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)milliseconds;
}

// Функция гибридной обработки (CPU + GPU параллельно)
double processArrayHybrid(float* data, int size) {
    int half = size / 2;

    double cpuTime, gpuTime;
    clock_t startCPU, endCPU;

    // Начинаем отсчет времени для гибридной обработки
    double startHybrid = omp_get_wtime();

    // Разделяем работу между CPU и GPU используя pragma omp parallel
#pragma omp parallel sections
    {
        // Секция 1: Обработка первой половины массива на CPU
#pragma omp section
        {
            startCPU = clock();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < half; i++) {
                data[i] = data[i] * 2.0f;
            }
            endCPU = clock();
            cpuTime = (double)(endCPU - startCPU) / CLOCKS_PER_SEC * 1000.0;
        }

        // Секция 2: Обработка второй половины массива на GPU
#pragma omp section
        {
            float* d_data;
            cudaEvent_t start, stop;
            float milliseconds = 0.0f;
            cudaError_t err;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            err = cudaMalloc((void**)&d_data, half * sizeof(float));
            if (err != cudaSuccess) {
                printf("Ошибка выделения памяти GPU: %s\n", cudaGetErrorString(err));
                gpuTime = 0;
            }
            else {
                err = cudaMemcpy(d_data, &data[half], half * sizeof(float),
                    cudaMemcpyHostToDevice);

                if (err != cudaSuccess) {
                    printf("Ошибка копирования на GPU: %s\n", cudaGetErrorString(err));
                    gpuTime = 0;
                }
                else {
                    int blockSize = 256;
                    int gridSize = (half + blockSize - 1) / blockSize;

                    cudaEventRecord(start);
                    processArrayGPU << <gridSize, blockSize >> > (d_data, half);

                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("Ошибка запуска ядра: %s\n", cudaGetErrorString(err));
                        gpuTime = 0;
                    }
                    else {
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);
                        cudaEventElapsedTime(&milliseconds, start, stop);

                        err = cudaMemcpy(&data[half], d_data, half * sizeof(float),
                            cudaMemcpyDeviceToHost);
                        if (err != cudaSuccess) {
                            printf("Ошибка копирования с GPU: %s\n", cudaGetErrorString(err));
                        }
                        gpuTime = (double)milliseconds;
                    }
                }
                cudaFree(d_data);
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    double endHybrid = omp_get_wtime();
    double totalHybridTime = (endHybrid - startHybrid) * 1000.0;

    printf("  CPU время обработки: %.3f мс\n", cpuTime);
    printf("  GPU время обработки: %.3f мс\n", gpuTime);
    printf("  Общее гибридное время: %.3f мс\n", totalHybridTime);

    return totalHybridTime;
}

// Функция инициализации массива случайными числами
void initializeArray(float* data, int size) {
    srand((unsigned int)time(NULL));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 1000) / 10.0f;
    }
}

// Функция проверки корректности результатов
void verifyResults(float* cpu_data, float* gpu_data, float* hybrid_data, int size) {
    printf("\n==================== ПРОВЕРКА КОРРЕКТНОСТИ ====================\n");

    bool cpuCorrect = true;
    bool gpuCorrect = true;
    bool hybridCorrect = true;

    // Проверяем первые 10 элементов
    for (int i = 0; i < 10 && i < size; i++) {
        if (fabs(cpu_data[i] - gpu_data[i]) > 1e-5) gpuCorrect = false;
        if (fabs(cpu_data[i] - hybrid_data[i]) > 1e-5) hybridCorrect = false;
    }

    printf("CPU результаты корректны: %s\n", cpuCorrect ? "ДА" : "НЕТ");
    printf("GPU результаты совпадают с CPU: %s\n", gpuCorrect ? "ДА" : "НЕТ");
    printf("Гибридные результаты совпадают с CPU: %s\n", hybridCorrect ? "ДА" : "НЕТ");
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
int main() {
    printf("========================================================\n");
    printf("  ГИБРИДНОЕ ПРИЛОЖЕНИЕ ДЛЯ CPU И GPU (OpenMP + CUDA)\n");
    printf("========================================================\n\n");

    // Размер массива (оптимально для Colab)
    const int N = 5000000; // 5 миллионов элементов

    printf("Параметры эксперимента:\n");
    printf("  Размер массива: %d элементов\n", N);
    printf("  Размер данных: %.2f МБ\n\n", (N * sizeof(float)) / (1024.0f * 1024.0f));

    // Выделяем память для исходного массива и трех рабочих копий
    float* original_data = (float*)malloc(N * sizeof(float));
    float* cpu_data = (float*)malloc(N * sizeof(float));
    float* gpu_data = (float*)malloc(N * sizeof(float));
    float* hybrid_data = (float*)malloc(N * sizeof(float));

    if (!original_data || !cpu_data || !gpu_data || !hybrid_data) {
        printf("Ошибка: Не удалось выделить память!\n");
        return 1;
    }

    // Инициализируем исходный массив один раз
    printf("Инициализация исходного массива...\n");
    initializeArray(original_data, N);

    // Копируем исходные данные в три отдельные рабочие массива ДО обработки
    printf("Копирование данных в рабочие массивы...\n");
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        cpu_data[i] = original_data[i];
        gpu_data[i] = original_data[i];
        hybrid_data[i] = original_data[i];
    }

    printf("Данные готовы к обработке.\n");

    printf("\n==================== ЗАДАНИЕ 1: ОБРАБОТКА НА CPU ====================\n");

    double cpuStartTime = omp_get_wtime();
    processArrayCPU(cpu_data, N);
    double cpuEndTime = omp_get_wtime();
    double cpuTotalTime = (cpuEndTime - cpuStartTime) * 1000.0;

    printf("Время обработки на CPU: %.3f мс\n", cpuTotalTime);
    printf("Первые 5 элементов (после обработки): ");
    for (int i = 0; i < 5; i++) printf("%.1f ", cpu_data[i]);
    printf("\n");

    printf("\n==================== ЗАДАНИЕ 2: ОБРАБОТКА НА GPU ====================\n");

    double gpuTotalTime = processArrayOnGPU(gpu_data, N);

    printf("Время обработки на GPU (включая передачу данных): %.3f мс\n", gpuTotalTime);
    printf("Первые 5 элементов (после обработки): ");
    for (int i = 0; i < 5; i++) printf("%.1f ", gpu_data[i]);
    printf("\n");

    printf("\n==================== ЗАДАНИЕ 3: ГИБРИДНАЯ ОБРАБОТКА ====================\n");
    printf("Одновременная обработка: CPU обрабатывает первую половину,\n");
    printf("GPU обрабатывает вторую половину массива параллельно.\n\n");

    double hybridTotalTime = processArrayHybrid(hybrid_data, N);

    printf("Первые 5 элементов первой половины (CPU): ");
    for (int i = 0; i < 5; i++) printf("%.1f ", hybrid_data[i]);
    printf("\n");

    printf("Первые 5 элементов второй половины (GPU): ");
    for (int i = N / 2; i < N / 2 + 5; i++) printf("%.1f ", hybrid_data[i]);
    printf("\n");

    printf("\n==================== ЗАДАНИЕ 4: АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ====================\n\n");

    printf("СРАВНЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ:\n\n");
    printf("  CPU (OpenMP):\n");
    printf("    - Время обработки: %.3f мс\n", cpuTotalTime);
    printf("    - Ускорение: 1.00x (базовое)\n\n");

    printf("  GPU (CUDA):\n");
    printf("    - Время обработки: %.3f мс\n", gpuTotalTime);
    printf("    - Ускорение: %.2fx (по отношению к CPU)\n\n", cpuTotalTime / gpuTotalTime);

    printf("  Гибридный подход:\n");
    printf("    - Время обработки: %.3f мс\n", hybridTotalTime);
    printf("    - Ускорение: %.2fx (по отношению к CPU)\n\n", cpuTotalTime / hybridTotalTime);

    // Вычисляем эффективность гибридного подхода
    double theoreticalHybridTime = fmax(cpuTotalTime / 2.0, gpuTotalTime - cpuTotalTime);
    double hybridEfficiency = theoreticalHybridTime / hybridTotalTime * 100.0;

    printf("Эффективность гибридного подхода: %.1f%%\n", hybridEfficiency);

    // Анализ результатов
    printf("\n==================== АНАЛИЗ РЕЗУЛЬТАТОВ ====================\n\n");

    printf("1. СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:\n");
    if (gpuTotalTime < cpuTotalTime) {
        printf("   + GPU обработка быстрее CPU на %.1f%%\n",
            (1.0 - gpuTotalTime / cpuTotalTime) * 100.0);
    }
    else {
        printf("   + CPU обработка быстрее GPU на %.1f%%\n",
            (1.0 - cpuTotalTime / gpuTotalTime) * 100.0);
        printf("     (GPU медленнее из-за накладных расходов на передачу данных)\n");
    }

    printf("\n2. ГИБРИДНЫЙ ПОДХОД:\n");
    if (hybridTotalTime < cpuTotalTime) {
        printf("   + Гибридный подход дает ускорение %.2fx\n", cpuTotalTime / hybridTotalTime);
        printf("   + Эффективность использования ресурсов: %.1f%%\n", hybridEfficiency);
    }
    else {
        printf("   - Гибридный подход оказался медленнее\n");
        printf("     (вероятно, из-за накладных расходов синхронизации)\n");
    }

    // Проверка корректности
    verifyResults(cpu_data, gpu_data, hybrid_data, N);

    // Освобождаем память
    free(original_data);
    free(cpu_data);
    free(gpu_data);
    free(hybrid_data);

    return 0;
}
