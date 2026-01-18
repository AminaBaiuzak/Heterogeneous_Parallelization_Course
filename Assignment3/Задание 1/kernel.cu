#include <cuda_runtime.h>  // Основные функции CUDA
#include <iostream>        // Для вывода в консоль
#include <chrono>          // Для замеров времени выполнения

#define ARRAY_SIZE 1000000  // Размер массива для обработки (1 миллион элементов)

// ---------------------------
// Версия 1: глобальная память
// ---------------------------
__global__ void multiply_global(float* arr, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Вычисляем глобальный индекс потока в массиве
    if (idx < n) {                                  // Проверяем, что поток не выходит за предел массива
        arr[idx] *= factor;                         // Умножаем элемент массива на factor
    }
}

// ---------------------------
// Версия 2: разделяемая память (shared memory)
// ---------------------------
__global__ void multiply_shared(float* arr, float factor, int n) {
    extern __shared__ float sdata[];               // Объявляем массив в shared memory для блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока

    if (idx < n)
        sdata[threadIdx.x] = arr[idx];             // Копируем данные из глобальной памяти в shared
    __syncthreads();                               // Ждем, пока все потоки блока скопируют данные

    if (idx < n)
        sdata[threadIdx.x] *= factor;              // Обработка элементов в shared memory
    __syncthreads();                               // Синхронизация перед записью обратно в глобальную память

    if (idx < n)
        arr[idx] = sdata[threadIdx.x];            // Записываем обработанные данные обратно в глобальную память
}

// ---------------------------
// Вспомогательная функция проверки ошибок CUDA
// ---------------------------
void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {                    // Если функция CUDA вернула ошибку
        std::cerr << "CUDA Error: " << msg         // Выводим сообщение об ошибке
            << " : " << cudaGetErrorString(result) << std::endl; // И подробное описание ошибки
        exit(EXIT_FAILURE);                        // Завершаем программу
    }
}

// ---------------------------
// Главная функция программы
// ---------------------------
int main() {
    int n = ARRAY_SIZE;                             // Размер массива
    size_t size = n * sizeof(float);               // Размер памяти в байтах для одного массива

    // Выделяем память на хосте (CPU)
    float* h_arr = new float[n];                    // Создаем массив на CPU
    for (int i = 0; i < n; ++i) h_arr[i] = 1.0f * i; // Инициализируем массив значениями от 0 до n-1

    // Выделяем память на устройстве (GPU)
    float* d_arr;                                   // Указатель на массив на GPU
    checkCuda(cudaMalloc(&d_arr, size), "Alloc d_arr"); // Выделяем память на GPU
    checkCuda(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice), "Memcpy H->D"); // Копируем данные с CPU на GPU

    int threadsPerBlock = 256;                      // Количество потоков в одном блоке
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Вычисляем количество блоков

    // ---------------------------
    // Тест глобальной памяти
    // ---------------------------
    auto start = std::chrono::high_resolution_clock::now(); // Запоминаем время начала
    multiply_global << <blocks, threadsPerBlock >> > (d_arr, 2.0f, n); // Запускаем kernel для глобальной памяти
    cudaDeviceSynchronize();                          // Ждем завершения всех потоков GPU
    auto end = std::chrono::high_resolution_clock::now(); // Фиксируем время конца
    std::cout << "Global Memory Time: "
        << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n"; // Выводим время выполнения

    // Сбрасываем данные на GPU для следующего теста
    checkCuda(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice), "Reset d_arr");

    // ---------------------------
    // Тест shared памяти
    // ---------------------------
    start = std::chrono::high_resolution_clock::now(); // Начало таймера
    multiply_shared << <blocks, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_arr, 2.0f, n);
    // Запускаем kernel с shared memory (размер shared = threadsPerBlock * sizeof(float))
    cudaDeviceSynchronize();                          // Синхронизируем потоки
    end = std::chrono::high_resolution_clock::now();  // Конец таймера
    std::cout << "Shared Memory Time: "
        << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n"; // Вывод времени

    // ---------------------------
    // Очистка памяти
    // ---------------------------
    cudaFree(d_arr);                                  // Освобождаем память на GPU
    delete[] h_arr;                                   // Освобождаем память на CPU

    return 0;                                        // Завершаем программу
}