#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <omp.h>  // OpenMP

int main() {
    setlocale(LC_ALL, "Russian"); // Устанавливает русскую локаль для программы, чтобы корректно отображать русские символы в консоли (например, вывод текста на русском языке)
    const int N = 10000;  // размер массива
    std::vector<int> arr(N);

    // Заполнение массива случайными числами
    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        arr[i] = std::rand();
    }

    // Последовательный поиск min/max
    int min_seq = std::numeric_limits<int>::max();
    int max_seq = std::numeric_limits<int>::min();

    double t1 = omp_get_wtime();  // время начала
    for (int i = 0; i < N; ++i) {
        if (arr[i] < min_seq) min_seq = arr[i];
        if (arr[i] > max_seq) max_seq = arr[i];
    }
    double t2 = omp_get_wtime();  // время конца
    double seq_time = (t2 - t1) * 1000; // в миллисекундах
    std::cout << "Sequential: min = " << min_seq
        << ", max = " << max_seq
        << ", time = " << seq_time << " ms\n";

    // Параллельный поиск min/max
    int min_par = std::numeric_limits<int>::max();
    int max_par = std::numeric_limits<int>::min();

    t1 = omp_get_wtime();  // время начала
#pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; ++i) {
        if (arr[i] < min_par) min_par = arr[i];
        if (arr[i] > max_par) max_par = arr[i];
    }
    t2 = omp_get_wtime();  // время конца
    double par_time = (t2 - t1) * 1000; // в миллисекундах
    std::cout << "Parallel  : min = " << min_par
        << ", max = " << max_par
        << ", time = " << par_time << " ms\n";

    // Выводы
    std::cout << "\nВыводы:\n";
    if (min_seq == min_par && max_seq == max_par)
        std::cout << "Оба метода дают одинаковые результаты (min и max совпадают).\n";
    if (seq_time < par_time)
        std::cout << "На маленьком массиве последовательная версия быстрее параллельной.\n";
    else
        std::cout << "Параллельная версия быстрее.\n";
    std::cout << "При увеличении размера массива эффективность параллельной версии возрастает.\n";

    return 0;
}
