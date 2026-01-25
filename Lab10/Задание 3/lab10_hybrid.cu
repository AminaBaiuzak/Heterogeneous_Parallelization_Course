#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

#define N 1024 * 1024 * 16  // 16 миллионов элементов
#define BLOCK_SIZE 256      // размер блока для GPU

// ==========================
// GPU-ядро: простое умножение на 2
// ==========================
__global__ void gpu_kernel(float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// ==========================
// Функция замеров времени для GPU
// ==========================
float run_gpu_kernel(float* d_in, float* d_out, int n, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    gpu_kernel << <grid, block, 0, stream >> > (d_in, d_out, n);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// ==========================
// Главная функция
// ==========================
int main() {
    std::cout << "Гибридная программа CPU + GPU" << std::endl;

    // ==========================
    // Инициализация данных на CPU
    // ==========================
    std::vector<float> h_in(N);
    std::vector<float> h_out(N, 0.0f);

    for (int i = 0; i < N; i++) h_in[i] = i % 100;

    // ==========================
    // Разделяем массив на две части:
    // первая половина на CPU, вторая на GPU
    // ==========================
    int split = N / 2;

    // ==========================
    // CPU: вычисления на первой половине
    // ==========================
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thread([&]() {
        for (int i = 0; i < split; i++) {
            h_out[i] = h_in[i] * 2.0f;
        }
        });

    // ==========================
    // GPU: вычисления на второй половине
    // ==========================
    float* d_in, * d_out;
    cudaMalloc(&d_in, split * sizeof(float));
    cudaMalloc(&d_out, split * sizeof(float));

    // Создаём поток для асинхронной передачи и вычислений
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Асинхронная передача данных на GPU
    auto transfer_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_in, h_in.data() + split, split * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Запуск GPU-ядра
    float gpu_time = run_gpu_kernel(d_in, d_out, split, stream);

    // Асинхронная копия результата обратно на CPU
    cudaMemcpyAsync(h_out.data() + split, d_out, split * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Ждём завершения CPU и GPU
    cpu_thread.join();
    cudaStreamSynchronize(stream);
    auto transfer_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpu_duration = std::chrono::high_resolution_clock::now() - cpu_start;
    std::chrono::duration<double> transfer_duration = transfer_end - cpu_start;

    std::cout << "Время CPU (пол-массива): " << cpu_duration.count() * 1000 << " мс" << std::endl;
    std::cout << "Общее время гибридного приложения: " << transfer_duration.count() * 1000 << " мс" << std::endl;
    std::cout << "Время GPU вычислений: " << gpu_time << " мс" << std::endl;

    // ==========================
    // Проверка корректности
    // ==========================
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i] * 2.0f) {
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Результат проверки: корректно!" << std::endl;
    else    std::cout << "Ошибка в вычислениях!" << std::endl;

    // ==========================
    // Очистка ресурсов
    // ==========================
    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    return 0;
}
