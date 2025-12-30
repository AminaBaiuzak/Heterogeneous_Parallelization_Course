#include <iostream>
#include <chrono>  
#include <cstdlib>  
#include <ctime>  
#include <omp.h>    

using clk = std::chrono::high_resolution_clock;

void chaeck_omp_support() {
    std::cout << "Maximum OpenMP threads available: " << omp_get_max_threads() << std::endl;

    #pragma omp parallel
    {
        #pragma omp master
        {
            int num_threads = omp_get_num_threads();
            std::cout << "Number of threads in parallel region: "
                << num_threads << std::endl << std::endl;
        }
    }
}

// Задание 1. Последовательное вычисление среднего
double compute_average(int* arr, size_t size) {
    auto start = clk::now();  // замер начала выполнения

    long long sum = 0;        // используем long long, чтобы избежать переполнения
    for (size_t i = 0; i < size; i++)
        sum += arr[i];        // суммируем все элементы массива

    double avg = static_cast<double>(sum) / size;  // вычисляем среднее

    auto end = clk::now();    // замер конца выполнения
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Average = " << avg << ", Time = " << duration_ms << " ms\n" << std::endl;

    return avg;
}

// Задание 2. Последовательный поиск min/max
void seq_minmax(int* arr, size_t size, int& min_val, int& max_val) {
    auto start = clk::now();

    min_val = arr[0];         // инициализация min первым элементом
    max_val = arr[0];         // инициализация max первым элементом
    for (size_t i = 1; i < size; i++) {
        if (arr[i] < min_val) min_val = arr[i];  // ищем минимум
        if (arr[i] > max_val) max_val = arr[i];  // ищем максимум
    }

    auto end = clk::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sequential Min = " << min_val << ", Max = " << max_val
        << ", Time = " << duration_ms << " ms\n" << std::endl;
}

// Задание 3. Параллельный поиск min/max с использованием OpenMP
void parallel_minmax(int* arr, size_t size, int& min_val, int& max_val) {
    auto start = clk::now();

    // Параллельный цикл с редукцией для min и max
    // reduction(min:min_val) — каждый поток считает локальный min и потом объединяет
    // reduction(max:max_val) — каждый поток считает локальный max и потом объединяет
    int local_min = arr[0];
    int local_max = arr[0];
    int num_threads = 0;


    // Внутри каждого потока выполняется редукция min/max
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < size; i++) {
        if (arr[i] < local_min) local_min = arr[i];
        if (arr[i] > local_max) local_max = arr[i];
    }

    // присваиваем результат внешним переменным после параллельного блока
    min_val = local_min;
    max_val = local_max;

    auto end = clk::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel Min = " << min_val << ", Max = " << max_val
        << ", Time = " << duration_ms << " ms" << std::endl << std::endl;
}

// Задание 4. Сравнение последовательного и параллельного вычисления среднего
void parallel_average(int* arr, size_t size) {
    // Последовательное вычисление
    auto start = clk::now();
    long long sum = 0;
    for (size_t i = 0; i < size; i++)
        sum += arr[i];
    double avg_seq = static_cast<double>(sum) / size;
    auto end = clk::now();
    double time_seq = std::chrono::duration<double, std::milli>(end - start).count();

    // Параллельное вычисление с редукцией
    start = clk::now();
    long long sum_par = 0;


    #pragma omp parallel for reduction(+:sum_par)  // каждый поток считает часть суммы, потом объединяет
    for (int i = 0; i < size; i++)
        sum_par += arr[i];

    double avg_par = static_cast<double>(sum_par) / size;
    end = clk::now();
    double time_par = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Sequential Avg = " << avg_seq << ", Time = " << time_seq << " ms\n";
    std::cout << "Parallel Avg = " << avg_par << ", Time = " << time_par << " ms\n";
}

int main() {

	chaeck_omp_support();  // проверка поддержки OpenMP

    std::srand(std::time(nullptr));  // инициализация генератора случайных чисел

    // Задание 1
    size_t size1 = 50000;
    int* arr1 = new int[size1];      // динамический массив
    for (size_t i = 0; i < size1; i++) arr1[i] = std::rand() % 100 + 1; // числа 1...100
    compute_average(arr1, size1);    // вызываем функцию вычисления среднего
    delete[] arr1;                   // освобождаем память

    // Задание 2 и 3
    size_t size2 = 1000000;
    int* arr2 = new int[size2];
    for (size_t i = 0; i < size2; i++) arr2[i] = std::rand() % 100000; // числа 0..99999

    int min_val, max_val;
    seq_minmax(arr2, size2, min_val, max_val);       // последовательный поиск
    parallel_minmax(arr2, size2, min_val, max_val);  // параллельный поиск
    delete[] arr2;

    // Задание 4
    size_t size4 = 5000000;
    int* arr4 = new int[size4];
    for (size_t i = 0; i < size4; i++) arr4[i] = std::rand() % 100000;
    parallel_average(arr4, size4);      // последовательное и параллельное среднее
    delete[] arr4;

    return 0;
}