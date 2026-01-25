#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Функция для проверки ошибок CUDA
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Ядро CUDA для вычисления суммы элементов в глобальной памяти
// Используем алгоритм редукции (уменьшения)
__global__ void sumArrayGlobalMemory(float* array, float* result, int size) {
    // Получаем глобальный индекс потока
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Используем разделяемую память внутри блока для частичных сумм
    extern __shared__ float sdata[];

    // Каждый поток загружает свой элемент в разделяемую память
    if (tid < size) {
        sdata[threadIdx.x] = array[tid];
    }
    else {
        sdata[threadIdx.x] = 0.0f;
    }

    // Синхронизация всех потоков в блоке
    __syncthreads();

    // Выполняем редукцию внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Первый поток каждого блока записывает результат
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Последовательная реализация на CPU
float sumArrayCPU(float* array, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    const int N = 100000;  // Размер массива
    const int blockSize = 256;  // Размер блока в CUDA
    const int gridSize = (N + blockSize - 1) / blockSize;  // Количество блоков

    // Выделение памяти на хосте (CPU)
    float* h_array = new float[N];
    float* h_result = new float[gridSize];

    // Инициализация массива случайными числами
    std::srand(42);
    for (int i = 0; i < N; i++) {
        h_array[i] = static_cast<float>(rand()) / RAND_MAX;  // Числа от 0 до 1
    }

    // Замер времени для CPU реализации
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_sum = sumArrayCPU(h_array, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Выделение памяти на устройстве (GPU)
    float* d_array = nullptr;
    float* d_result = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_array, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, gridSize * sizeof(float)));

    // Копирование данных с хоста на устройство
    CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice));

    // Замер времени для GPU реализации
    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Запуск ядра CUDA
    sumArrayGlobalMemory << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_array, d_result, N);

    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    // Измерение времени выполнения ядра
    float gpu_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, end));

    // Копирование результата обратно на хост
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Финальная сумма результатов всех блоков
    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        gpu_sum += h_result[i];
    }

    // Вывод результатов
    std::cout << "=== ЗАДАНИЕ 1: Сумма элементов массива ===" << std::endl;
    std::cout << "Размер массива: " << N << " элементов" << std::endl;
    std::cout << "CPU результат: " << cpu_sum << std::endl;
    std::cout << "GPU результат: " << gpu_sum << std::endl;
    std::cout << "Разница: " << std::abs(cpu_sum - gpu_sum) << std::endl;
    std::cout << "Время CPU: " << cpu_time.count() * 1000 << " мс" << std::endl;
    std::cout << "Время GPU: " << gpu_time_ms << " мс" << std::endl;
    std::cout << "Ускорение: " << cpu_time.count() * 1000 / gpu_time_ms << "x" << std::endl;


    // Освобождение памяти
    delete[] h_array;
    delete[] h_result;
    CHECK_CUDA_ERROR(cudaFree(d_array));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));

    return 0;
}