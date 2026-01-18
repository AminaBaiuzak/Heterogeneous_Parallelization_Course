#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <locale>  // для setlocale
#include <clocale> // для setlocale

int main() {
    std::setlocale(LC_ALL, "Russian");
    // Размеры матриц
    const int N = 4;  // строки A
    const int M = 3;  // столбцы A / строки B
    const int K = 5;  // столбцы B

    // 1. Создаем матрицы A, B и C
    std::vector<float> A(N * M);
    std::vector<float> B(M * K);
    std::vector<float> C(N * K, 0.0f); // результат OpenCL

    // Заполняем A и B простыми числами
    for (int i = 0; i < N * M; ++i) A[i] = i + 1;      // 1,2,3...
    for (int i = 0; i < M * K; ++i) B[i] = 1.0f;      // все единицы

    // 2. Последовательное умножение на CPU (для проверки)
    std::vector<float> C_seq(N * K, 0.0f);
    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * K + j];
            }
            C_seq[i * K + j] = sum;
        }
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_seq = end_seq - start_seq;
    std::cout << "CPU (последовательное) время: " << duration_seq.count() << " ms\n";

    // 3. Получаем платформу и устройство
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    for (cl_uint p = 0; p < platformCount; ++p) {
        cl_platform_id platform = platforms[p];

        cl_uint deviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        for (cl_uint d = 0; d < deviceCount; ++d) {
            cl_device_id device = devices[d];

            // 4. Создаем контекст и очередь
            cl_int err;
            cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            cl_queue_properties props[] = { 0 };
            cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

            // 5. Загружаем и компилируем ядро
            std::ifstream kernelFile("matrix_mul.cl");
            std::ostringstream oss;
            oss << kernelFile.rdbuf();
            std::string srcStrStd = oss.str();
            const char* srcStr = srcStrStd.c_str();

            cl_program program = clCreateProgramWithSource(context, 1, &srcStr, nullptr, &err);
            err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                // Выводим лог компиляции
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                std::cout << "Build log:\n" << log.data() << std::endl;
            }

            cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);

            // 6. Создаем буферы
            cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * A.size(), A.data(), &err);
            cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * B.size(), B.data(), &err);
            cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                sizeof(float) * C.size(), nullptr, &err);

            // 7. Передаем аргументы ядра
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
            clSetKernelArg(kernel, 3, sizeof(int), &N);
            clSetKernelArg(kernel, 4, sizeof(int), &M);
            clSetKernelArg(kernel, 5, sizeof(int), &K);

            // 8. Настраиваем глобальные размеры (N x K)
            size_t globalSize[2] = { (size_t)N, (size_t)K };
            // локальная группа (опционально, можно nullptr)
            size_t localSize[2] = { 16, 16 };

            // 9. Запуск ядра и замер времени
            auto start = std::chrono::high_resolution_clock::now();
            clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
            clFinish(queue);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;

            // 10. Считываем результат
            clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);

            // 11. Выводим результат
            std::cout << "Platform " << p << " Device " << d
                << " время: " << duration.count() << " ms\nC = \n";
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < K; ++j) std::cout << C[i * K + j] << " ";
                std::cout << std::endl;
            }

            // 12. Проверка совпадения с CPU
            bool correct = true;
            for (int i = 0; i < N * K; ++i) {
                if (std::fabs(C[i] - C_seq[i]) > 1e-5) correct = false;
            }
            if (correct) std::cout << "Результат верный!\n\n";
            else std::cout << "Ошибка в вычислениях!\n";

            // 13. Освобождаем ресурсы
            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }

    return 0;
}
