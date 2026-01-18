#include <CL/cl.h>           // OpenCL API
#include <iostream>          // Для вывода в консоль
#include <vector>            // Для работы с динамическими массивами
#include <chrono>            // Для замера времени
#include <fstream>           // Для чтения ядра из файла
#include <sstream>           // Для потоковой работы с текстом

int main() {
    const int N = 1000000; // Размер массивов
    std::vector<float> A(N, 1.0f); // Массив A, заполнен 1.0
    std::vector<float> B(N, 2.0f); // Массив B, заполнен 2.0
    std::vector<float> C(N, 0.0f); // Массив C для результата

    // -------------------------------
    // Получаем количество доступных OpenCL платформ
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount); // Сначала узнаём количество платформ
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr); // Получаем платформы

    std::cout << "OpenCL platforms found: " << platformCount << std::endl;

    // -------------------------------
    // Перебираем все платформы
    for (cl_uint p = 0; p < platformCount; ++p) {
        cl_platform_id platform = platforms[p];

        cl_uint deviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount); // Считаем устройства
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr); // Получаем устройства

        // Перебираем устройства платформы
        for (cl_uint d = 0; d < deviceCount; ++d) {
            cl_device_id device = devices[d];

            // -------------------------------
            // Определяем тип устройства: CPU или GPU
            cl_device_type deviceType;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr);
            std::string typeName = (deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" :
                (deviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";
            std::cout << "\nPlatform " << p << " Device " << d << " is " << typeName << std::endl;

            // -------------------------------
            // Создаём контекст для устройства
            cl_int err;
            cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cout << "Error creating context: " << err << std::endl;
                continue;
            }

            // -------------------------------
            // Создаём очередь команд (новый метод)
            cl_queue_properties props[] = { 0 }; // Пустые свойства
            cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
            if (err != CL_SUCCESS) {
                std::cout << "Error creating command queue: " << err << std::endl;
                clReleaseContext(context);
                continue;
            }

            // -------------------------------
            // Загружаем ядро из файла kernel.cl
            std::ifstream kernelFile("kernel.cl");
            if (!kernelFile.is_open()) {
                std::cout << "Failed to open kernel.cl" << std::endl;
                return 1;
            }
            std::ostringstream oss;
            oss << kernelFile.rdbuf();
            std::string srcStdStr = oss.str();
            const char* srcStr = srcStdStr.c_str();

            // -------------------------------
            // Создаём программу OpenCL
            cl_program program = clCreateProgramWithSource(context, 1, &srcStr, nullptr, &err);
            if (err != CL_SUCCESS) { std::cout << "Error creating program: " << err << std::endl; }

            // Компилируем программу
            err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                // Если ошибка компиляции, выводим лог
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                std::cout << "Build log:\n" << log.data() << std::endl;
            }

            // Создаём объект ядра
            cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
            if (err != CL_SUCCESS) { std::cout << "Error creating kernel: " << err << std::endl; }

            // -------------------------------
            // Создаём буферы для массивов
            cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * N, A.data(), &err);
            cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * N, B.data(), &err);
            cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, &err);

            // -------------------------------
            // Передаём аргументы ядра
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

            // -------------------------------
            // Запускаем ядро и замеряем время
            size_t globalSize = N; // количество элементов
            auto start = std::chrono::high_resolution_clock::now();
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
            clFinish(queue); // ждем окончания выполнения
            auto end = std::chrono::high_resolution_clock::now();

            // Считываем результат
            clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * N, C.data(), 0, nullptr, nullptr);

            // -------------------------------
            // Выводим время выполнения для текущего устройства
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

            // Проверяем первые 5 элементов массива
            std::cout << "C[0..4] = ";
            for (int i = 0; i < 5; ++i) std::cout << C[i] << " ";
            std::cout << std::endl;

            // -------------------------------
            // Освобождаем ресурсы OpenCL
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
