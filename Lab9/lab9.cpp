#include <mpi.h>              // Подключение библиотеки MPI для параллельных вычислений
#include <iostream>           // Для ввода/вывода в консоль
#include <vector>             // Для использования динамических массивов (std::vector)
#include <cmath>              // Для математических функций, например sqrt
#include <cstdlib>            // Для функций rand, srand
#include <ctime>              // Для получения текущего времени (используется для srand)
#include <iomanip>            // Для форматирования вывода (setprecision, setw)
#include <algorithm>          // Для стандартных алгоритмов (например, max, min)

using namespace std;          // Чтобы не писать std:: перед cout, vector и т.д.


// ===========================================
// ЗАДАНИЕ 1: Распределённое вычисление среднего значения и стандартного отклонения
// ===========================================
void task1_mean_stddev(int rank, int size) {
    const int N = 1000000;                   // Размер массива для вычислений
    vector<double> data;                     // Вектор для исходных данных (только у процесса 0)
    vector<int> sendcounts(size);            // Массив с количеством элементов, которые отправляем каждому процессу
    vector<int> displs(size);                // Массив смещений (откуда брать элементы для каждого процесса)

    // Инициализация данных только на процессе 0
    if (rank == 0) {
        cout << "\n=== ЗАДАНИЕ 1: Вычисление среднего значения и стандартного отклонения ===" << endl;
        cout << "Размер массива: " << N << endl;
        cout << "Количество процессов: " << size << endl;

        data.resize(N);                      // Выделяем память под массив
        srand(time(NULL));                    // Инициализация генератора случайных чисел
        for (int i = 0; i < N; i++) {        // Заполняем массив случайными числами от 0 до 100
            data[i] = (double)rand() / RAND_MAX * 100.0;
        }
    }

    // Вычисление, сколько элементов получит каждый процесс
    int base_count = N / size;               // Базовое количество элементов на процесс
    int remainder = N % size;                // Остаток элементов, которые нужно распределить

    for (int i = 0; i < size; i++) {
        // Процессы с индексом < remainder получают по одному элементу больше
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        // Смещение для Scatterv: откуда брать данные для процесса
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_count = sendcounts[rank];      // Количество элементов для текущего процесса
    vector<double> local_data(local_count);  // Вектор для локальных данных процесса

    // Рассылаем части массива data всем процессам
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_data.data(), local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0.0;                  // Локальная сумма значений
    double local_sum_squares = 0.0;          // Локальная сумма квадратов (для дисперсии)

    // Вычисляем локальную сумму и сумму квадратов
    for (int i = 0; i < local_count; i++) {
        local_sum += local_data[i];
        local_sum_squares += local_data[i] * local_data[i];
    }

    double global_sum = 0.0;                 // Глобальная сумма
    double global_sum_squares = 0.0;         // Глобальная сумма квадратов

    // Суммируем локальные суммы всех процессов на процессе 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_squares, &global_sum_squares, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Главный процесс вычисляет среднее и стандартное отклонение
    if (rank == 0) {
        double mean = global_sum / N;        // Среднее значение
        double variance = (global_sum_squares / N) - (mean * mean);  // Дисперсия
        double stddev = sqrt(variance);      // Стандартное отклонение

        cout << fixed << setprecision(6);    // Форматируем вывод
        cout << "Среднее значение: " << mean << endl;
        cout << "Стандартное отклонение: " << stddev << endl;
    }
}


// ===========================================
// ЗАДАНИЕ 2: Распределённое решение системы линейных уравнений методом Гаусса
// ===========================================
void task2_gaussian_elimination(int rank, int size) {
    const int N = 4;                          // Размер системы
    vector<double> A;                          // Вектор для хранения расширенной матрицы (N x (N+1))
    vector<double> x(N);                       // Вектор решения

    // Инициализация матрицы только на процессе 0
    if (rank == 0) {
        cout << "\n=== ЗАДАНИЕ 2: Решение системы линейных уравнений методом Гаусса ===" << endl;
        cout << "Размер системы: " << N << "x" << N << endl;

        A.resize(N * (N + 1));                 // Выделяем память под расширенную матрицу

        double matrix[4][5] = {                // Исходная расширенная матрица (4x5)
            {2, 1, -1, 1, 5},
            {3, 2, 2, -1, 8},
            {1, 3, -1, 2, 9},
            {4, -1, 3, 1, 10}
        };

        // Копируем матрицу в вектор A
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N + 1; j++) {
                A[i * (N + 1) + j] = matrix[i][j];
            }
        }

        // Вывод исходной матрицы
        cout << "Исходная расширенная матрица:" << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N + 1; j++) {
                cout << setw(10) << A[i * (N + 1) + j] << " ";
            }
            cout << endl;
        }
    }

    vector<double> pivot_row(N + 1);           // Вектор для хранения ведущей строки

    // Прямой ход метода Гаусса
    for (int k = 0; k < N; k++) {
        if (rank == 0) {                       // Главный процесс копирует ведущую строку
            for (int j = 0; j < N + 1; j++) {
                pivot_row[j] = A[k * (N + 1) + j];
            }
        }

        // Рассылаем ведущую строку всем процессам
        MPI_Bcast(pivot_row.data(), N + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Обработка строк (на данном примере выполняется только процессом 0)
        for (int i = k + 1 + rank; i < N; i += size) {
            if (rank == 0) {
                double factor = A[i * (N + 1) + k] / pivot_row[k]; // Коэффициент для исключения переменной
                for (int j = k; j < N + 1; j++) {
                    A[i * (N + 1) + j] -= factor * pivot_row[j]; // Вычитание ведущей строки
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);            // Синхронизация процессов
    }

    // Обратный ход для нахождения решения
    if (rank == 0) {
        for (int i = N - 1; i >= 0; i--) {
            x[i] = A[i * (N + 1) + N];
            for (int j = i + 1; j < N; j++) {
                x[i] -= A[i * (N + 1) + j] * x[j];
            }
            x[i] /= A[i * (N + 1) + i];
        }

        // Вывод решения
        cout << "\nРешение системы:" << endl;
        for (int i = 0; i < N; i++) {
            cout << "x[" << i << "] = " << setprecision(6) << x[i] << endl;
        }
    }
}


// ===========================================
// ЗАДАНИЕ 3: Параллельный анализ графов (алгоритм Флойда-Уоршелла)
// ===========================================
void task3_floyd_warshall(int rank, int size) {
    const int N = 5;                           // Количество вершин графа
    vector<double> graph;                       // Вектор для матрицы смежности
    const double INF = 1e9;                     // Значение "бесконечность" для отсутствующих рёбер

    // Инициализация графа только на процессе 0
    if (rank == 0) {
        cout << "\n=== ЗАДАНИЕ 3: Поиск кратчайших путей (алгоритм Флойда-Уоршелла) ===" << endl;
        cout << "Размер графа: " << N << " вершин" << endl;

        graph.resize(N * N);                    // Выделяем память под матрицу смежности

        double adj[5][5] = {                     // Исходная матрица смежности
            {0, 3, INF, 7, INF},
            {8, 0, 2, INF, INF},
            {5, INF, 0, 1, INF},
            {2, INF, INF, 0, 4},
            {INF, INF, INF, 6, 0}
        };

        // Копируем матрицу в вектор graph
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                graph[i * N + j] = adj[i][j];
            }
        }

        // Вывод исходной матрицы смежности
        cout << "Исходная матрица смежности:" << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (graph[i * N + j] >= INF) {
                    cout << setw(8) << "INF";
                }
                else {
                    cout << setw(8) << graph[i * N + j];
                }
            }
            cout << endl;
        }
    }

    // Определяем количество строк на каждый процесс
    int rows_per_proc = N / size;
    int extra_rows = N % size;
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    vector<double> local_graph(local_rows * N); // Вектор для локальных строк
    vector<int> sendcounts(size);               // Количество элементов для Scatterv
    vector<int> displs(size);                   // Смещения для Scatterv

    // Заполняем sendcounts и displs
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * N;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    // Рассылаем строки графа каждому процессу
    MPI_Scatterv(graph.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_graph.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> k_row(N);                     // Вектор для k-й строки (для алгоритма Флойда)

    // Основной цикл алгоритма Флойда
    for (int k = 0; k < N; k++) {
        int k_owner = 0;                        // Процесс, который владеет k-й строкой
        int cumulative = 0;
        for (int p = 0; p < size; p++) {
            int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
            if (k < cumulative + p_rows) {
                k_owner = p;
                break;
            }
            cumulative += p_rows;
        }

        // Копируем k-ю строку локально на процессе-владельце
        if (rank == k_owner) {
            int local_k = k - displs[rank] / N;
            for (int j = 0; j < N; j++) {
                k_row[j] = local_graph[local_k * N + j];
            }
        }

        // Рассылаем k-ю строку всем процессам
        MPI_Bcast(k_row.data(), N, MPI_DOUBLE, k_owner, MPI_COMM_WORLD);

        // Обновляем локальные строки графа
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < N; j++) {
                double new_dist = local_graph[i * N + k] + k_row[j];
                if (local_graph[i * N + j] > new_dist) {
                    local_graph[i * N + j] = new_dist;
                }
            }
        }
    }

    // Собираем обновлённые строки обратно на процесс 0
    MPI_Gatherv(local_graph.data(), local_rows * N, MPI_DOUBLE,
        graph.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Главный процесс выводит результат
    if (rank == 0) {
        cout << "\nМатрица кратчайших путей:" << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (graph[i * N + j] >= INF) {
                    cout << setw(8) << "INF";
                }
                else {
                    cout << setw(8) << graph[i * N + j];
                }
            }
            cout << endl;
        }
    }
}


// ===========================================
// ГЛАВНАЯ ФУНКЦИЯ
// ===========================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);                     // Инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Получаем номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Получаем общее количество процессов

    // Вывод информации о работе (только на процессе 0)
    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "Практическая работа №9: MPI" << endl;
        cout << "========================================" << endl;
    }

    // ===========================================
    // Задание 1
    // ===========================================
    double start_time = MPI_Wtime();            // Замер времени
    task1_mean_stddev(rank, size);              // Выполнение задания 1
    double end_time = MPI_Wtime();              // Конец замера

    if (rank == 0) {
        cout << "Время выполнения задания 1: " << end_time - start_time << " секунд" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);                // Синхронизация процессов

    // ===========================================
    // Задание 2
    // ===========================================
    start_time = MPI_Wtime();
    task2_gaussian_elimination(rank, size);
    end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Время выполнения задания 2: " << end_time - start_time << " секунд" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);                // Синхронизация процессов

    // ===========================================
    // Задание 3
    // ===========================================
    start_time = MPI_Wtime();
    task3_floyd_warshall(rank, size);
    end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Время выполнения задания 3: " << end_time - start_time << " секунд" << endl;
    }

    MPI_Finalize();                             // Завершение работы MPI
    return 0;                                   // Выход из программы
}
