#include <cuda_runtime.h>  // Основные функции CUDA
#include <iostream>        // Для вывода в консоль
#include <chrono>          // Для замеров времени

#define ARRAY_SIZE 1000000  // Размер массива (1 миллион элементов)

// ---------------------------
// Kernel для сложения двух массивов
// ---------------------------
__global__ void add_arrays(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Вычисляем глобальный индекс потока
    if (idx < n) {                                  // Проверяем, что поток не выходит за предел массива
        c[idx] = a[idx] + b[idx];                   // Сложение элементов a[idx] + b[idx] → c[idx]
    }
}

// ---------------------------
// Вспомогательная функция проверки ошибок CUDA
// ---------------------------
void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {                     // Если функция CUDA вернула ошибку
        std::cerr << "CUDA Error: " << msg          // Выводим сообщение об ошибке
            << " : " << cudaGetErrorString(result) << std::endl; // Подробное описание ошибки
        exit(EXIT_FAILURE);                         // Завершаем программу
    }
}

// ---------------------------
// Главная функция программы
// ---------------------------
int main() {
    int n = ARRAY_SIZE;                             // Размер массива
    size_t size = n * sizeof(float);                // Размер памяти в байтах

    // ---------------------------
    // Выделение памяти на CPU
    // ---------------------------
    float* h_a = new float[n];                       // Первый массив на CPU
    float* h_b = new float[n];                       // Второй массив на CPU
    float* h_c = new float[n];                       // Результирующий массив на CPU

    // Инициализация массивов
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f * i;                           // Значения от 0 до n-1
        h_b[i] = 2.0f * i;                           // Значения от 0 до 2*(n-1)
        h_c[i] = 0.0f;                               // Результат обнуляем
    }

    // ---------------------------
    // Выделение памяти на GPU
    // ---------------------------
    float* d_a, * d_b, * d_c;
    checkCuda(cudaMalloc(&d_a, size), "Alloc d_a"); // Память для a
    checkCuda(cudaMalloc(&d_b, size), "Alloc d_b"); // Память для b
    checkCuda(cudaMalloc(&d_c, size), "Alloc d_c"); // Память для c

    // Копирование данных с CPU на GPU
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Memcpy h_a -> d_a");
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Memcpy h_b -> d_b");

    // ---------------------------
    // Тестирование с разными размерами блоков
    // ---------------------------
    int block_sizes[3] = { 128, 256, 512 };           // Три размера блоков для сравнения

    for (int bs : block_sizes) {                     // Цикл по размерам блоков
        int blocks = (n + bs - 1) / bs;             // Количество блоков для текущего размера

        auto start = std::chrono::high_resolution_clock::now(); // Таймер начало
        add_arrays << <blocks, bs >> > (d_a, d_b, d_c, n);          // Запуск kernel
        cudaDeviceSynchronize();                                 // Синхронизация потоков
        auto end = std::chrono::high_resolution_clock::now();   // Таймер конец

        // Вывод времени выполнения
        std::cout << "Block Size " << bs << ": "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms" << std::endl;
    }

    // ---------------------------
    // Копирование результата обратно на CPU (необязательно для замеров)
    // ---------------------------
    checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "Memcpy d_c -> h_c");

    // ---------------------------
    // Очистка памяти
    // ---------------------------
    cudaFree(d_a);                                   // Освобождение GPU памяти
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;                                    // Освобождение CPU памяти
    delete[] h_b;
    delete[] h_c;

    return 0;                                        // Завершение программы
}