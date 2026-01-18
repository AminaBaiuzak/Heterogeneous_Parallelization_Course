%% writefile task3_coalesced.cu
#include <stdio.h>             // Для printf
#include <cuda_runtime.h>      // Основные функции CUDA Runtime API
#include <time.h>              // Для работы с временем (не используется в этом варианте, можно убрать)

// ---------------------------
// Константы
// ---------------------------
#define ARRAY_SIZE 1000000                     // Размер массива для тестирования
#define BLOCK_SIZE 256                         // Количество потоков в одном блоке
#define GRID_SIZE (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE  // Количество блоков в сетке (округление вверх)

// ---------------------------
// Макрос для проверки ошибок CUDA
// ---------------------------
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// КОАЛЕСЦИРОВАННЫЙ ДОСТУП
// Каждый поток обрабатывает последовательный элемент массива
__global__ void kernel_coalesced(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // Вычисляем глобальный индекс потока

    if (idx < size) {                                  // Проверяем, что поток не выходит за границы массива
        // Коалесцированный доступ:
        // Потоки 0,1,2,... обращаются к data[0], data[1], data[2], ...
        data[idx] = data[idx] * 2.0f + 1.0f;          // Умножаем на 2 и прибавляем 1
    }
}

// НЕКОАЛЕСЦИРОВАННЫЙ ДОСТУП
// Потоки обращаются к элементам с "разбросанным" шагом
__global__ void kernel_uncoalesced(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс потока

    if (idx < size) {                                 // Проверяем, что поток не выходит за массив
        // Вычисляем индекс с шагом (stride pattern) для разрозненного доступа
        int stride_idx = (idx % BLOCK_SIZE) * GRID_SIZE + (idx / BLOCK_SIZE);

        if (stride_idx < size) {                       // Проверка на выход за предел массива
            data[stride_idx] = data[stride_idx] * 2.0f + 1.0f;  // Модификация элемента
        }
    }
}

// Функция для замера времени выполнения ядра на GPU
float measure_kernel_time(void (*kernel)(float*, int), float* d_data, int size, int iterations) {
    cudaEvent_t start, stop;                           // Создаём события CUDA для замера времени
    CHECK_CUDA(cudaEventCreate(&start));               // Создание события start
    CHECK_CUDA(cudaEventCreate(&stop));                // Создание события stop

    // Разогрев GPU (чтобы исключить первый запуск)
    kernel << <GRID_SIZE, BLOCK_SIZE >> > (d_data, size);
    CHECK_CUDA(cudaDeviceSynchronize());               // Ждём завершения kernel

    // Замер времени
    CHECK_CUDA(cudaEventRecord(start));                // Старт замера
    for (int i = 0; i < iterations; i++) {            // Повторяем kernel несколько раз для точного замера
        kernel << <GRID_SIZE, BLOCK_SIZE >> > (d_data, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));                 // Фиксируем конец
    CHECK_CUDA(cudaEventSynchronize(stop));           // Ждём окончания всех операций

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));  // Вычисляем прошедшее время

    CHECK_CUDA(cudaEventDestroy(start));              // Освобождаем событие start
    CHECK_CUDA(cudaEventDestroy(stop));               // Освобождаем событие stop

    return milliseconds / iterations;                 // Возвращаем среднее время одного запуска kernel
}

// Главная функция
int main() {
    printf("===============================================\n");
    printf("CUDA: Коалесцированный vs Некоалесцированный доступ\n");
    printf("===============================================\n\n");

    // Получение информации об устройстве
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));    // Считываем свойства GPU 0
    printf("GPU: %s\n", prop.name);                   // Выводим название GPU
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);  // Версия вычислительных возможностей
    printf("Max Block Size: %d\n", prop.maxThreadsPerBlock);        // Максимальное количество потоков в блоке
    printf("Warp Size: %d\n\n", prop.warpSize);                       // Размер warp'a (32 потока)

    // Вывод параметров теста
    printf("Параметры теста:\n");
    printf("  Размер массива: %d элементов\n", ARRAY_SIZE);
    printf("  Размер блока: %d потоков\n", BLOCK_SIZE);
    printf("  Размер сетки: %d блоков\n", GRID_SIZE);
    printf("  Элементов на блок: %d\n\n", BLOCK_SIZE);

    // Выделение памяти
    float* d_data_coalesced = nullptr;                 // Указатель для коалесцированного массива на GPU
    float* d_data_uncoalesced = nullptr;               // Указатель для некоалесцированного массива на GPU
    float* h_data = nullptr;                           // Массив на CPU

    CHECK_CUDA(cudaMalloc(&d_data_coalesced, ARRAY_SIZE * sizeof(float))); // Выделение памяти на GPU
    CHECK_CUDA(cudaMalloc(&d_data_uncoalesced, ARRAY_SIZE * sizeof(float)));
    h_data = (float*)malloc(ARRAY_SIZE * sizeof(float));                   // Выделение памяти на CPU

    // Инициализация данных на CPU
    printf("Инициализация данных...\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)i * 0.1f;                    // Заполняем массив числами 0.0,0.1,0.2,...
    }

    // Копируем данные на GPU
    CHECK_CUDA(cudaMemcpy(d_data_coalesced, h_data, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_data_uncoalesced, h_data, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int iterations = 100;                                // Количество повторов для замера времени

    // ---------------------------
    // Вывод заголовка результатов
    // ---------------------------
    printf("\n===============================================\n");
    printf("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n");
    printf("===============================================\n\n");

    // ---------------------------
    // Тест коалесцированного доступа
    // ---------------------------
    printf("1. КОАЛЕСЦИРОВАННЫЙ ДОСТУП:\n");
    printf("   Потоки обращаются к последовательным адресам памяти\n");
    float time_coalesced = measure_kernel_time(kernel_coalesced, d_data_coalesced, ARRAY_SIZE, iterations); // Замер
    printf("   Время выполнения: %.4f мс\n", time_coalesced);
    printf("   Пропускная способность: %.2f GB/s\n",
        (ARRAY_SIZE * sizeof(float) * 2 / 1e9) / (time_coalesced / 1000)); // Вычисляем скорость передачи данных

    printf("\n");

    // ---------------------------
    // Тест некоалесцированного доступа
    // ---------------------------
    printf("2. НЕКОАЛЕСЦИРОВАННЫЙ ДОСТУП:\n");
    printf("   Потоки обращаются к элементам с шагом (stride pattern)\n");
    float time_uncoalesced = measure_kernel_time(kernel_uncoalesced, d_data_uncoalesced, ARRAY_SIZE, iterations);
    printf("   Время выполнения: %.4f мс\n", time_uncoalesced);
    printf("   Пропускная способность: %.2f GB/s\n",
        (ARRAY_SIZE * sizeof(float) * 2 / 1e9) / (time_uncoalesced / 1000));

    printf("\n");
    printf("===============================================\n");
    printf("АНАЛИЗ РЕЗУЛЬТАТОВ\n");
    printf("===============================================\n");
    printf("Разница во времени: %.4f мс\n", time_uncoalesced - time_coalesced);
    printf("Ускорение коалесцированного доступа: %.2f x\n", time_uncoalesced / time_coalesced);
    printf("Замедление некоалесцированного доступа: %.1f %%\n",
        ((time_uncoalesced / time_coalesced) - 1) * 100);

    // ---------------------------
    // Очистка памяти
    // ---------------------------
    CHECK_CUDA(cudaFree(d_data_coalesced));
    CHECK_CUDA(cudaFree(d_data_uncoalesced));
    free(h_data);

    printf("\n===============================================\n");

    return 0;
}
