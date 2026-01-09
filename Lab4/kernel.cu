#include <iostream>  // для ввода/вывода
#include <cstdlib>   // для стандартных функций C (например, rand)
#include <chrono>    // для измерения времени выполнения
#include <cuda.h>    // основной заголовок для работы с CUDA

using namespace std;

/* =========================================================
   ЗАДАНИЕ 1. Генерация массива случайных чисел
   ========================================================= */

   // __global__ — это спецификатор CUDA для функции, которая выполняется на GPU
   // и может быть вызвана с использованием <<<blocks, threads>>> (параллельно многими потоками)
__global__ void generate_array(int* arr, int n) {
    // Рассчитываем уникальный индекс потока в глобальной сетке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за пределы массива
    if (idx < n) {
        arr[idx] = idx % 100; // заполняем массив числами от 0 до 99
        // Можно заменить на случайное число: arr[idx] = rand() % 100;
    }
}

/* =========================================================
   ЗАДАНИЕ 2a. Редукция (глобальная память)
   ========================================================= */

__global__ void reduction_global(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока

    if (idx < n) {
        // атомарная операция сложения в глобальной памяти
        // гарантирует, что несколько потоков не испортят результат
        atomicAdd(output, input[idx]);
    }
}

/* =========================================================
   ЗАДАНИЕ 2b. Редукция (глобальная + разделяемая память)
   ========================================================= */

__global__ void reduction_shared(int* input, int* output, int n) {
    __shared__ int cache[256]; // выделяем быструю разделяемую память на блок

    int tid = threadIdx.x; // индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid; // глобальный индекс потока

    // копируем данные в shared memory для ускорения вычислений
    if (idx < n) cache[tid] = input[idx];
    else cache[tid] = 0; // если поток выходит за массив, кладём 0

    __syncthreads(); // ждём, пока все потоки блока закончат копирование

    // редукция в shared memory (суммирование элементов внутри блока)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { // делим пополам шаг за шагом
        if (tid < s)
            cache[tid] += cache[tid + s]; // складываем с соответствующим элементом
        __syncthreads(); // синхронизация потоков на каждом шаге
    }

    // поток с tid = 0 добавляет сумму блока в глобальную память
    if (tid == 0)
        atomicAdd(output, cache[0]);
}

/* =========================================================
   ЗАДАНИЕ 3a. Пузырьковая сортировка подмассивов (локальная память)
   ========================================================= */

__global__ void bubble_sort_subarrays(int* data, int sub_size) {
    int block = blockIdx.x;         // номер блока = номер подмассива
    int start = block * sub_size;   // начало подмассива в общем массиве

    int local[32]; // локальная память (регистры) для подмассива
    // сюда копируем данные из глобальной памяти для быстрой работы

    for (int i = 0; i < sub_size; i++)
        local[i] = data[start + i];

    // сортировка пузырьком: проходим подмассив несколько раз
    for (int i = 0; i < sub_size - 1; i++) {
        for (int j = 0; j < sub_size - i - 1; j++) {
            if (local[j] > local[j + 1]) {
                // меняем элементы местами
                int tmp = local[j];
                local[j] = local[j + 1];
                local[j + 1] = tmp;
            }
        }
    }

    // записываем отсортированные данные обратно в глобальную память
    for (int i = 0; i < sub_size; i++)
        data[start + i] = local[i];
}

/* =========================================================
   ЗАДАНИЕ 3b. Слияние подмассивов (разделяемая память)
   ========================================================= */

__global__ void merge_subarrays(int* data, int* temp, int sub_size) {
    __shared__ int shared[64]; // shared memory для ускорения слияния

    int block = blockIdx.x;
    int start = block * 2 * sub_size; // начало двух подмассивов для слияния

    // копируем два подмассива в shared memory
    for (int i = threadIdx.x; i < 2 * sub_size; i += blockDim.x) {
        shared[i] = data[start + i];
    }
    __syncthreads(); // ждём, пока все потоки скопируют данные

    // поток 0 блока выполняет слияние
    if (threadIdx.x == 0) {
        int i = 0, j = sub_size, k = start;
        // стандартный алгоритм слияния
        while (i < sub_size && j < 2 * sub_size)
            temp[k++] = (shared[i] < shared[j]) ? shared[i++] : shared[j++];
        while (i < sub_size) temp[k++] = shared[i++];
        while (j < 2 * sub_size) temp[k++] = shared[j++];
    }
}

/* =========================================================
   ЗАДАНИЕ 4. Главная функция с измерением времени
   ========================================================= */

int main() {

    int sizes[3] = { 10000, 100000, 1000000 }; // размеры массивов для теста

    // перебираем все размеры массивов
    for (int s = 0; s < 3; s++) {

        int N = sizes[s];
        cout << "\n=========================================\n";
        cout << "Размер массива: " << N << " элементов\n";

        // выделяем память на GPU для массива
        int* d_array;
        cudaMalloc(&d_array, N * sizeof(int));

        // ------------------------
        // Генерация массива на GPU
        // ------------------------
        // каждый поток заполняет один элемент массива
        generate_array << <(N + 255) / 256, 256 >> > (d_array, N);

        // ------------------------
        // Редукция: глобальная память
        // ------------------------
        int* d_sum;
        cudaMalloc(&d_sum, sizeof(int)); // память для суммы
        cudaMemset(d_sum, 0, sizeof(int)); // обнуляем перед вычислением

        auto start = chrono::high_resolution_clock::now(); // старт таймера
        reduction_global << <(N + 255) / 256, 256 >> > (d_array, d_sum, N);
        cudaDeviceSynchronize(); // ждём, пока все потоки закончат
        auto end = chrono::high_resolution_clock::now(); // стоп таймера

        cout << "Время редукции (глобальная память): "
            << chrono::duration<double, milli>(end - start).count() << " мс\n";

        // ------------------------
        // Редукция: разделяемая память
        // ------------------------
        cudaMemset(d_sum, 0, sizeof(int));
        start = chrono::high_resolution_clock::now();
        reduction_shared << <(N + 255) / 256, 256 >> > (d_array, d_sum, N);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();

        cout << "Время редукции (разделяемая память): "
            << chrono::duration<double, milli>(end - start).count() << " мс\n";

        // ------------------------
        // Сортировка пузырьком подмассивов
        // ------------------------
        int sub_size = 32; // размер подмассива
        int numBlocks = (N + sub_size - 1) / sub_size; // количество блоков подмассивов

        int* d_temp; // временный массив для слияния
        cudaMalloc(&d_temp, N * sizeof(int));

        start = chrono::high_resolution_clock::now();
        bubble_sort_subarrays << <numBlocks, 1 >> > (d_array, sub_size);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();

        cout << "Время пузырьковой сортировки подмассивов: "
            << chrono::duration<double, milli>(end - start).count() << " мс\n";

        // ------------------------
        // Слияние подмассивов
        // ------------------------
        int mergeBlocks = (N + 2 * sub_size - 1) / (2 * sub_size);
        start = chrono::high_resolution_clock::now();
        merge_subarrays << <mergeBlocks, 64 >> > (d_array, d_temp, sub_size);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();

        cout << "Время слияния подмассивов: "
            << chrono::duration<double, milli>(end - start).count() << " мс\n";

        // ------------------------
        // Освобождаем память на GPU
        // ------------------------
        cudaFree(d_array);
        cudaFree(d_sum);
        cudaFree(d_temp);
    }

    return 0; // успешное завершение программы
}
