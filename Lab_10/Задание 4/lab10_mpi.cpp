#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>      // для std::accumulate
#include <cstdlib>      // для std::rand
#include <chrono>       // для замеров времени

#define N 10000000       // 10 миллионов элементов

int main(int argc, char* argv[]) {
    // ==========================
    // Инициализация MPI
    // ==========================
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // общее число процессов

    if (rank == 0) {
        std::cout << "MPI программа с " << size << " процессами" << std::endl;
    }

    // ==========================
    // Подготовка данных
    // ==========================
    int local_N = N / size;  // количество элементов на процесс

    std::vector<int> local_data(local_N);

    // Заполняем массив случайными числами (каждый процесс независимо)
    std::srand(rank + 1);  // разное зерно для каждого процесса
    for (int i = 0; i < local_N; i++) {
        local_data[i] = std::rand() % 100;  // числа 0..99
    }

    // ==========================
    // Локальная обработка (CPU)
    // ==========================
    auto local_start = std::chrono::high_resolution_clock::now();

    long long local_sum = std::accumulate(local_data.begin(), local_data.end(), 0LL);

    auto local_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> local_duration = local_end - local_start;

    std::cout << "Процесс " << rank << ": локальная сумма = " << local_sum
        << ", время локальной обработки = " << local_duration.count() * 1000 << " мс" << std::endl;

    // ==========================
    // Глобальная операция MPI_Reduce
    // ==========================
    long long global_sum = 0;

    auto reduce_start = std::chrono::high_resolution_clock::now();

    // MPI_Reduce суммирует локальные значения и отправляет результат процессу 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    auto reduce_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reduce_duration = reduce_end - reduce_start;

    if (rank == 0) {
        std::cout << "Глобальная сумма элементов массива = " << global_sum << std::endl;
        std::cout << "Время MPI_Reduce = " << reduce_duration.count() * 1000 << " мс" << std::endl;
    }

    // ==========================
    // Альтернатива: MPI_Allreduce
    // ==========================
    long long global_sum_all = 0;
    auto allreduce_start = std::chrono::high_resolution_clock::now();

    // MPI_Allreduce суммирует значения и возвращает результат всем процессам
    MPI_Allreduce(&local_sum, &global_sum_all, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    auto allreduce_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> allreduce_duration = allreduce_end - allreduce_start;

    if (rank == 0) {
        std::cout << "Глобальная сумма с MPI_Allreduce = " << global_sum_all << std::endl;
        std::cout << "Время MPI_Allreduce = " << allreduce_duration.count() * 1000 << " мс" << std::endl;
    }

    // ==========================
    // Завершение MPI
    // ==========================
    MPI_Finalize();
    return 0;
}
