#include <iostream>         
#include <vector>          
#include <cstdlib>          
#include <ctime>            
#include <chrono>           // Для измерения времени выполнения
#include <omp.h>            // Для директив OpenMP

// Последовательный Selection Sort
void selection_sort_seq(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {       // Проходим по всем элементам массива
        int min_idx = i;                     // Считаем текущий элемент минимальным
        for (int j = i + 1; j < n; ++j) {   // Ищем минимальный элемент в остатке массива
            if (arr[j] < arr[min_idx]) {
                min_idx = j;                 // Обновляем индекс минимального элемента
            }
        }
        std::swap(arr[i], arr[min_idx]);     // Меняем текущий элемент с минимальным найденным
    }
}

// Параллельный Selection Sort с OpenMP 
// Мы параллельно ищем минимальный элемент на каждом шаге.
// Важно использовать критическую секцию для обновления global_min_idx
void selection_sort_parallel(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        int global_min_idx = i;
        int global_min = arr[i];

#pragma omp parallel   // Параллельный регион
        {
            int local_min = global_min;
            int local_min_idx = global_min_idx;

#pragma omp for nowait   // Параллельный for без ожидания в конце
            for (int j = i + 1; j < n; ++j) {
                if (arr[j] < local_min) {
                    local_min = arr[j];
                    local_min_idx = j;
                }
            }

#pragma omp critical    // Критическая секция для безопасного обновления глобального минимума
            {
                if (local_min < global_min) {
                    global_min = local_min;
                    global_min_idx = local_min_idx;
                }
            }
        }

        std::swap(arr[i], arr[global_min_idx]); // Меняем текущий элемент с минимальным найденным
    }
}

// Бенчмарк
void benchmark(int n) {
    // Создаем массив случайных чисел размером n
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = rand();

    // Создаем копии массива для последовательной и параллельной сортировки
    auto seq_arr = arr;
    auto par_arr = arr;

    // Измеряем время последовательной сортировки
    auto t1 = std::chrono::high_resolution_clock::now(); // Засекаем время старта
    selection_sort_seq(seq_arr);                  // Последовательная сортировка выбором
    auto t2 = std::chrono::high_resolution_clock::now(); // Засекаем время окончания
    double seq_time = std::chrono::duration<double, std::milli>(t2 - t1).count(); // Вычисляем время в миллисекундах

    // Измеряем время параллельной сортировки
    t1 = std::chrono::high_resolution_clock::now();     // Засекаем время старта
    selection_sort_parallel(par_arr);                   // Параллельная сортировка выбором с OpenMP
    t2 = std::chrono::high_resolution_clock::now();    // Засекаем время окончания
    double par_time = std::chrono::duration<double, std::milli>(t2 - t1).count(); // Вычисляем время в миллисекундах

    // Вывод результатов сортировки
    std::cout << "\nРазмер массива: " << n << "\n";
    std::cout << "Sequential sort time: " << seq_time << " ms\n"; // Время последовательной сортировки
    std::cout << "Parallel sort time  : " << par_time << " ms\n"; // Время параллельной сортировки

    // Выводы о производительности
    std::cout << "Вывод: ";
    if (par_time < seq_time) {
        // Если параллельная сортировка быстрее, сообщаем это
        std::cout << "Параллельная сортировка быстрее для массива размером " << n << ".\n";
    }
    else {
        // Если последовательная сортировка быстрее, объясняем возможную причину
        std::cout << "Последовательная сортировка быстрее для массива размером " << n
            << ", накладные расходы на параллельность превышают выигрыш.\n";
    }
}


int main() {
    setlocale(LC_ALL, "Russian"); // Устанавливает русскую локаль для программы
    srand(static_cast<unsigned int>(time(nullptr))); // Инициализация генератора случайных чисел

    // Проверка производительности для двух размеров массива
    benchmark(1000);
    benchmark(10000);

    return 0;
}


