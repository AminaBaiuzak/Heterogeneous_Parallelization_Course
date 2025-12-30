#include <iostream>     // Для ввода и вывода (std::cout)
#include <cstdlib>      // Для rand(), srand()
#include <ctime>        // Для time()
#include <omp.h>        // Для OpenMP
#include <chrono>       // Для измерения времени

// --------------------------------------------------
// Функция заполнения массива случайными числами
// --------------------------------------------------
void fill_array(int* arr, int size) {
    // Проходим по всем элементам массива
    for (int i = 0; i < size; i++) {
        // Генерируем случайное число от 1 до 100
        arr[i] = std::rand() % 100 + 1;
    }
}

// --------------------------------------------------
// Последовательное вычисление среднего значения
// --------------------------------------------------
double average_sequential(int* arr, int size) {
    long long sum = 0;   // Используем long long, чтобы избежать переполнения

    // Суммируем элементы массива
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return static_cast<double>(sum) / size;
}

// --------------------------------------------------
// Параллельное вычисление среднего значения (OpenMP)
// --------------------------------------------------
double average_parallel(int* arr, int size) {
    long long sum = 0;   // Общая переменная для суммы

    // Параллельный цикл
    // reduction(+:sum) означает:
    // каждый поток считает свою часть суммы,
    // затем все частичные суммы объединяются
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return static_cast<double>(sum) / size;
}

// --------------------------------------------------
// Главная функция
// --------------------------------------------------
int main() {
    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Размер массива
    int size = 1000000;

    // --------------------------------------------------
    // 1. Динамическое выделение памяти
    // --------------------------------------------------
    int* arr = new int[size];  // Создаём динамический массив

    // Заполняем массив случайными числами
    fill_array(arr, size);

    // --------------------------------------------------
    // 2. Последовательное вычисление среднего
    // --------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    double avg_seq = average_sequential(arr, size);
    auto end = std::chrono::high_resolution_clock::now();

    double time_seq =
        std::chrono::duration<double, std::milli>(end - start).count();

    // --------------------------------------------------
    // 3. Параллельное вычисление среднего (OpenMP)
    // --------------------------------------------------
    start = std::chrono::high_resolution_clock::now();
    double avg_par = average_parallel(arr, size);
    end = std::chrono::high_resolution_clock::now();

    double time_par =
        std::chrono::duration<double, std::milli>(end - start).count();

    // --------------------------------------------------
    // Вывод результатов
    // --------------------------------------------------
    std::cout << "Sequential average: " << avg_seq
        << ", Time: " << time_seq << " ms\n";

    std::cout << "Parallel average:   " << avg_par
        << ", Time: " << time_par << " ms\n";

    std::cout << "Threads used: "
        << omp_get_max_threads() << "\n";

    // --------------------------------------------------
    // 4. Освобождение памяти
    // --------------------------------------------------
    delete[] arr;   // Освобождаем динамическую память

    return 0;
}
