#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err) \
                      << " в " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA ядро для обработки массива (вычисление: sqrt(x) * 2 + sin(x))
__global__ void processArrayKernel(const float* input, float* output, int n) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем границы массива
    if (idx < n) {
        // Выполняем вычислительную работу
        float x = input[idx];
        output[idx] = sqrtf(x) * 2.0f + sinf(x);
    }
}

// Функция обработки массива на CPU
void processArrayCPU(const float* input, float* output, int start, int end) {
    // Каждый элемент обрабатывается независимо
    for (int i = start; i < end; i++) {
        float x = input[i];
        output[i] = std::sqrt(x) * 2.0f + std::sin(x);
    }
}

// Функция обработки массива на GPU
void processArrayGPU(const float* h_input, float* h_output, int n) {
    float* d_input, * d_output; // Указатели на память GPU
    size_t bytes = n * sizeof(float);

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Настраиваем параметры запуска: 256 потоков на блок
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Запускаем ядро CUDA
    processArrayKernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, n);

    // Проверяем ошибки
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результаты обратно
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Полная обработка только на CPU
void fullCPUProcessing(const float* input, float* output, int n) {
    processArrayCPU(input, output, 0, n);
}

// Полная обработка только на GPU
void fullGPUProcessing(const float* input, float* output, int n) {
    processArrayGPU(input, output, n);
}

// Гибридная обработка: часть на CPU, часть на GPU
void hybridProcessing(const float* input, float* output, int n, float cpuRatio = 0.3f) {
    // Определяем точку разделения массива
    int splitPoint = static_cast<int>(n * cpuRatio);

    // Создаем временные массивы для GPU части
    float* gpu_output = new float[n - splitPoint];

    // Запускаем CPU обработку в отдельном потоке
    std::thread cpuThread([&]() {
        processArrayCPU(input, output, 0, splitPoint);
        });

    // Параллельно запускаем GPU обработку
    processArrayGPU(input + splitPoint, gpu_output, n - splitPoint);

    // Ждем завершения CPU обработки
    cpuThread.join();

    // Копируем результаты GPU обработки в выходной массив
    for (int i = 0; i < n - splitPoint; i++) {
        output[splitPoint + i] = gpu_output[i];
    }

    // Освобождаем временную память
    delete[] gpu_output;
}

// Функция для проверки корректности результатов
bool verifyResults(const float* a, const float* b, int n, float tolerance = 1e-3f) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > tolerance) {
            if (errors < 5) { // Показываем только первые 5 ошибок
                std::cerr << "Несоответствие на позиции " << i
                    << ": a=" << a[i] << ", b=" << b[i]
                    << " (разница: " << diff << ")" << std::endl;
            }
            errors++;
        }
    }
    if (errors > 5) {
        std::cerr << "... и еще " << (errors - 5) << " ошибок" << std::endl;
    }
    return errors == 0;
}

int main() {
    const int N = 10000000; // Размер массива (10 миллионов элементов)

    // Выделяем память на CPU
    float* h_input = new float[N];
    float* h_output_cpu = new float[N];
    float* h_output_gpu = new float[N];
    float* h_output_hybrid = new float[N];

    // Инициализируем входной массив
    std::cout << "=== Задание 3: Гибридная обработка CPU+GPU ===" << std::endl << std::endl;
    std::cout << "Инициализация массива..." << std::endl;
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i % 1000 + 1); // Значения от 1 до 1000
    }
    std::cout << "Массив инициализирован" << std::endl << std::endl;

    std::cout << "Параметры:" << std::endl;
    std::cout << "  Размер массива: " << N << " элементов" << std::endl;
    std::cout << "  Размер данных: " << (N * sizeof(float) / (1024.0 * 1024.0)) << " МБ" << std::endl;
    std::cout << "  Операция: sqrt(x) * 2 + sin(x)" << std::endl << std::endl;

    // ==================== ТОЛЬКО CPU ====================
    std::cout << "[1] Обработка только на CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    fullCPUProcessing(h_input, h_output_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "    Время: " << cpu_time.count() << " мс" << std::endl << std::endl;

    // ==================== ТОЛЬКО GPU ====================
    std::cout << "[2] Обработка только на GPU..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    fullGPUProcessing(h_input, h_output_gpu, N);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "    Время: " << gpu_time.count() << " мс" << std::endl << std::endl;

    // ==================== ГИБРИДНАЯ ОБРАБОТКА ====================
    std::cout << "[3] Гибридная обработка (30% CPU, 70% GPU)..." << std::endl;
    auto start_hybrid = std::chrono::high_resolution_clock::now();
    hybridProcessing(h_input, h_output_hybrid, N, 0.3f);
    auto end_hybrid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hybrid_time = end_hybrid - start_hybrid;
    std::cout << "    Время: " << hybrid_time.count() << " мс" << std::endl << std::endl;

    // ==================== ПРОВЕРКА КОРРЕКТНОСТИ ====================
    std::cout << "=== Проверка корректности ===" << std::endl;
    bool gpu_correct = verifyResults(h_output_cpu, h_output_gpu, N);
    bool hybrid_correct = verifyResults(h_output_cpu, h_output_hybrid, N);

    if (gpu_correct) {
        std::cout << "+ GPU результаты корректны" << std::endl;
    }
    else {
        std::cout << "- GPU результаты содержат ошибки" << std::endl;
    }

    if (hybrid_correct) {
        std::cout << "+ Гибридные результаты корректны" << std::endl;
    }
    else {
        std::cout << "- Гибридные результаты содержат ошибки" << std::endl;
    }
    std::cout << std::endl;

    // ==================== АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ====================
    std::cout << "=== Анализ производительности ===" << std::endl;
    std::cout << "CPU:       " << cpu_time.count() << " мс (базовая линия)" << std::endl;
    std::cout << "GPU:       " << gpu_time.count() << " мс";
    if (gpu_time.count() < cpu_time.count()) {
        std::cout << " (ускорение: " << (cpu_time.count() / gpu_time.count()) << "x) ";
    }
    else {
        std::cout << " (медленнее в " << (gpu_time.count() / cpu_time.count()) << "x)";
    }
    std::cout << std::endl;

    std::cout << "Гибридная: " << hybrid_time.count() << " мс";
    if (hybrid_time.count() < cpu_time.count()) {
        std::cout << " (ускорение: " << (cpu_time.count() / hybrid_time.count()) << "x)";
    }
    else {
        std::cout << " (медленнее в " << (hybrid_time.count() / cpu_time.count()) << "x)";
    }
    std::cout << std::endl << std::endl;

    // Определяем лучший метод
    double min_time = cpu_time.count();
    std::string best_method = "CPU";

    if (gpu_time.count() < min_time) {
        min_time = gpu_time.count();
        best_method = "GPU";
    }
    if (hybrid_time.count() < min_time) {
        min_time = hybrid_time.count();
        best_method = "Гибридная";
    }

    std::cout << "Самый быстрый метод: " << best_method << " (" << min_time << " мс)" << std::endl << std::endl;

    // ==================== ПОДРОБНЫЙ АНАЛИЗ ====================
    std::cout << "=== Подробный анализ ===" << std::endl;

    // Эффективность распределения работы
    int cpu_elements = static_cast<int>(N * 0.3f);
    int gpu_elements = N - cpu_elements;

    std::cout << "Распределение работы в гибридном режиме:" << std::endl;
    std::cout << "  CPU обработал: " << cpu_elements << " элементов (30%)" << std::endl;
    std::cout << "  GPU обработал: " << gpu_elements << " элементов (70%)" << std::endl << std::endl;

    // Пропускная способность
    double cpu_throughput = (N * sizeof(float) / (1024.0 * 1024.0)) / (cpu_time.count() / 1000.0);
    double gpu_throughput = (N * sizeof(float) / (1024.0 * 1024.0)) / (gpu_time.count() / 1000.0);
    double hybrid_throughput = (N * sizeof(float) / (1024.0 * 1024.0)) / (hybrid_time.count() / 1000.0);

    std::cout << "Пропускная способность:" << std::endl;
    std::cout << "  CPU:       " << cpu_throughput << " МБ/с" << std::endl;
    std::cout << "  GPU:       " << gpu_throughput << " МБ/с" << std::endl;
    std::cout << "  Гибридная: " << hybrid_throughput << " МБ/с" << std::endl << std::endl;

    // Показываем примеры результатов
    std::cout << "=== Примеры результатов ===" << std::endl;
    std::cout << "Первые 5 элементов:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] Вход: " << h_input[i]
            << " -> Выход: " << h_output_hybrid[i] << std::endl;
    }

    std::cout << "\nПоследние 3 элемента:" << std::endl;
    for (int i = N - 3; i < N; i++) {
        std::cout << "  [" << i << "] Вход: " << h_input[i]
            << " -> Выход: " << h_output_hybrid[i] << std::endl;
    }
    std::cout << std::endl;


    // Освобождаем память
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_output_hybrid;


    return 0;
}