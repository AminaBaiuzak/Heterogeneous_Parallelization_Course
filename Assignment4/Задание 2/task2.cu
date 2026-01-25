#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Функция для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err) \
                      << " в " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Последовательная версия на CPU для сравнения
void prefixSumCPU(const int* input, int* output, int n) {
    output[0] = input[0]; // Первый элемент остается таким же
    // Каждый следующий элемент - это сумма текущего и всех предыдущих
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// Оптимизированное CUDA ядро - сканирование Kogge-Stone внутри блока
// Это быстрый параллельный алгоритм для префиксной суммы
__global__ void blockScanKernel(int* data, int* blockSums, int n) {
    // Выделяем разделяемую память для блока (быстрая память на GPU)
    extern __shared__ int temp[];

    int tid = threadIdx.x; // Локальный индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс

    // Загружаем данные из глобальной памяти в разделяемую
    temp[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads(); // Ждем, пока все потоки загрузят данные

    // Выполняем параллельное сканирование (алгоритм Kogge-Stone)
    // На каждом шаге удваиваем расстояние между суммируемыми элементами
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride]; // Берем элемент на расстоянии stride
        }
        __syncthreads(); // Синхронизация перед обновлением

        if (tid >= stride) {
            temp[tid] += val; // Добавляем к текущему элементу
        }
        __syncthreads(); // Синхронизация после обновления
    }

    // Записываем результат обратно в глобальную память
    if (idx < n) {
        data[idx] = temp[tid];
    }

    // Последний поток каждого блока сохраняет сумму блока
    if (blockSums != nullptr && tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = temp[tid];
    }
}

// Ядро для добавления смещений к каждому блоку
__global__ void addBlockSums(int* data, int* blockSums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый блок (кроме первого) добавляет сумму всех предыдущих блоков
    if (idx < n && blockIdx.x > 0) {
        data[idx] += blockSums[blockIdx.x - 1];
    }
}

// Функция для запуска GPU вычислений
void prefixSumGPU(const int* h_input, int* h_output, int n) {
    int* d_input;  // Входные данные на GPU
    int* d_output; // Выходные данные на GPU
    size_t bytes = n * sizeof(int);

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Копируем входные данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, h_input, bytes, cudaMemcpyHostToDevice));

    // Настройка параметров запуска
    int threadsPerBlock = 256; // Количество потоков в блоке
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Количество блоков
    int sharedMemSize = threadsPerBlock * sizeof(int); // Размер разделяемой памяти

    // Массив для хранения сумм каждого блока
    int* d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));

    // ШАГ 1: Вычисляем префиксную сумму внутри каждого блока
    blockScanKernel << <numBlocks, threadsPerBlock, sharedMemSize >> > (d_output, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ШАГ 2: Вычисляем префиксную сумму сумм блоков
    if (numBlocks > 1) {
        // Копируем суммы блоков на CPU
        int* h_blockSums = new int[numBlocks];
        CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));

        // Вычисляем префиксную сумму на CPU (это быстро, т.к. блоков мало)
        for (int i = 1; i < numBlocks; i++) {
            h_blockSums[i] += h_blockSums[i - 1];
        }

        // Копируем обратно на GPU
        CUDA_CHECK(cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_blockSums;

        // ШАГ 3: Добавляем смещения к каждому блоку
        addBlockSums << <numBlocks, threadsPerBlock >> > (d_output, d_blockSums, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Копируем результаты обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_blockSums));
}

// Функция для проверки корректности результатов
bool verifyResults(const int* cpu, const int* gpu, int n, int maxErrors = 10) {
    int errorCount = 0;
    for (int i = 0; i < n; i++) {
        if (cpu[i] != gpu[i]) {
            if (errorCount < maxErrors) {
                std::cerr << "Несоответствие на позиции " << i
                    << ": CPU=" << cpu[i] << ", GPU=" << gpu[i] << std::endl;
            }
            errorCount++;
        }
    }
    if (errorCount > maxErrors) {
        std::cerr << "... и еще " << (errorCount - maxErrors) << " ошибок" << std::endl;
    }
    return errorCount == 0;
}

int main() {
    const int N = 1000000; // Размер массива

    // Выделяем память на CPU
    int* h_input = new int[N];
    int* h_output_cpu = new int[N];
    int* h_output_gpu = new int[N];

    // Инициализируем входной массив (каждый элемент = 1 для простоты)
    std::cout << "=== Задание 2: Префиксная сумма на CUDA ===" << std::endl << std::endl;
    std::cout << "Инициализация массива..." << std::endl;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1; // Простые данные: префиксная сумма даст 1,2,3,4...
    }
    std::cout << "Массив инициализирован" << std::endl << std::endl;

    std::cout << "Параметры:" << std::endl;
    std::cout << "  Размер массива: " << N << " элементов" << std::endl;
    std::cout << "  Размер данных: " << (N * sizeof(int) / (1024.0 * 1024.0)) << " МБ" << std::endl << std::endl;

    // ==================== ВЫЧИСЛЕНИЕ НА CPU ====================
    std::cout << "[1] Вычисление на CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_input, h_output_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "    Время: " << cpu_time.count() << " мс" << std::endl << std::endl;

    // ==================== ВЫЧИСЛЕНИЕ НА GPU ====================
    std::cout << "[2] Вычисление на GPU..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    prefixSumGPU(h_input, h_output_gpu, N);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "    Время: " << gpu_time.count() << " мс" << std::endl << std::endl;

    // ==================== ПРОВЕРКА КОРРЕКТНОСТИ ====================
    std::cout << "=== Проверка корректности ===" << std::endl;
    if (verifyResults(h_output_cpu, h_output_gpu, N)) {
        std::cout << "+ Результаты совпадают!" << std::endl;
    }
    else {
        std::cout << "- Обнаружены расхождения в результатах" << std::endl;
    }
    std::cout << std::endl;

    // ==================== АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ====================
    std::cout << "=== Анализ производительности ===" << std::endl;
    double speedup = cpu_time.count() / gpu_time.count();

    std::cout << "Время CPU:  " << cpu_time.count() << " мс" << std::endl;
    std::cout << "Время GPU:  " << gpu_time.count() << " мс" << std::endl;

    if (speedup > 1.0) {
        std::cout << "\nРезультат: GPU быстрее в " << speedup << " раз" << std::endl;
    }
    else if (speedup > 0.5) {
        std::cout << "\nРезультат: GPU и CPU показывают сопоставимую производительность" << std::endl;
        std::cout << "Коэффициент: " << speedup << "x" << std::endl;
    }
    else {
        std::cout << "\nРезультат: CPU быстрее в " << (1.0 / speedup) << " раз" << std::endl;
        std::cout << "Коэффициент: " << speedup << "x" << std::endl;
    }
    std::cout << std::endl;


    // Освобождаем память
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;


    return 0;
}
