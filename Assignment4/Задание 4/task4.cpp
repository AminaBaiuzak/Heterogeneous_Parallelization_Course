#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// Функция обработки локальной части массива
// Вычисляем сумму квадратов элементов
void processLocalArray(const std::vector<float>& local_data, float& local_result) {
    local_result = 0.0f;
    // Каждый процесс обрабатывает свою часть массива
    for (size_t i = 0; i < local_data.size(); i++) {
        local_result += local_data[i] * local_data[i]; // Сумма квадратов
    }
}

int main(int argc, char** argv) {
    // Инициализируем MPI окружение
    MPI_Init(&argc, &argv);

    int world_size; // Общее количество процессов
    int world_rank; // Ранг (номер) текущего процесса

    // Получаем количество процессов
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Получаем ранг текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int ARRAY_SIZE = 10000000; // Размер массива (10 миллионов)
    std::vector<float> global_data; // Весь массив (только на процессе 0)

    // Вычисляем размер локальной части для каждого процесса
    int local_size = ARRAY_SIZE / world_size;
    std::vector<float> local_data(local_size); // Локальная часть массива

    double start_time, end_time; // Переменные для измерения времени

    // ==================== ПРОЦЕСС 0 (ГЛАВНЫЙ) ====================
    if (world_rank == 0) {
        std::cout << "=== Распределённая обработка массива с MPI ===" << std::endl;
        std::cout << "Количество процессов: " << world_size << std::endl;
        std::cout << "Размер массива: " << ARRAY_SIZE << " элементов" << std::endl;
        std::cout << "Размер части для каждого процесса: " << local_size << " элементов" << std::endl;
        std::cout << "Размер данных: " << (ARRAY_SIZE * sizeof(float) / (1024.0 * 1024.0))
            << " МБ" << std::endl << std::endl;

        // Инициализируем глобальный массив на главном процессе
        global_data.resize(ARRAY_SIZE);
        std::cout << "Инициализация массива на главном процессе..." << std::endl;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            global_data[i] = static_cast<float>(i % 1000 + 1); // Значения от 1 до 1000
        }
        std::cout << "Инициализация завершена." << std::endl << std::endl;
    }

    // Синхронизируем все процессы перед началом измерений
    MPI_Barrier(MPI_COMM_WORLD);

    // Начинаем отсчет времени (на всех процессах)
    start_time = MPI_Wtime();

    // ==================== РАСПРЕДЕЛЕНИЕ ДАННЫХ ====================
    // Процесс 0 рассылает части массива всем остальным процессам
    // MPI_Scatter разделяет массив на равные части и отправляет каждому процессу
    MPI_Scatter(
        global_data.data(),     // Буфер отправки (только на процессе 0)
        local_size,              // Количество элементов для отправки каждому процессу
        MPI_FLOAT,               // Тип данных элементов
        local_data.data(),       // Буфер приема
        local_size,              // Количество элементов для приема
        MPI_FLOAT,               // Тип данных элементов
        0,                       // Ранг отправителя (главный процесс)
        MPI_COMM_WORLD           // Коммуникатор
    );

    if (world_rank == 0) {
        std::cout << "Данные распределены между процессами." << std::endl;
    }

    // ==================== ЛОКАЛЬНАЯ ОБРАБОТКА ====================
    // Каждый процесс независимо обрабатывает свою часть массива
    float local_result = 0.0f;
    processLocalArray(local_data, local_result);

    if (world_rank == 0) {
        std::cout << "Локальная обработка завершена на всех процессах." << std::endl;
    }

    // ==================== СБОР РЕЗУЛЬТАТОВ ====================
    // Собираем локальные результаты на главном процессе
    float global_result = 0.0f;

    // MPI_Reduce выполняет операцию суммирования всех локальных результатов
    MPI_Reduce(
        &local_result,           // Буфер отправки (локальный результат)
        &global_result,          // Буфер приема (только на процессе 0)
        1,                       // Количество элементов
        MPI_FLOAT,               // Тип данных
        MPI_SUM,                 // Операция (суммирование)
        0,                       // Ранг получателя (главный процесс)
        MPI_COMM_WORLD           // Коммуникатор
    );

    // Синхронизируем все процессы после завершения работы
    MPI_Barrier(MPI_COMM_WORLD);

    // Останавливаем отсчет времени
    end_time = MPI_Wtime();

    // ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
    // Только главный процесс выводит итоговые результаты
    if (world_rank == 0) {
        std::cout << "Результаты собраны на главном процессе." << std::endl << std::endl;

        std::cout << "=== Результаты вычислений ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Сумма квадратов всех элементов: " << global_result << std::endl;
        std::cout << "Время выполнения: " << (end_time - start_time) * 1000.0
            << " мс" << std::endl << std::endl;

        // Выводим информацию о вкладе каждого процесса
        std::cout << "=== Детали распределения ===" << std::endl;
        std::cout << "Каждый процесс обработал: " << local_size << " элементов" << std::endl;
        std::cout << "Процент данных на процесс: "
            << (100.0 * local_size / ARRAY_SIZE) << "%" << std::endl << std::endl;

        // Вычисляем теоретическую производительность
        double data_size_mb = (ARRAY_SIZE * sizeof(float)) / (1024.0 * 1024.0);
        double throughput = data_size_mb / ((end_time - start_time));
        std::cout << "Пропускная способность: " << throughput << " МБ/с" << std::endl;
    }
    else {
        // Остальные процессы могут вывести информацию о своей работе
        std::cout << "Процесс " << world_rank << " обработал " << local_size
            << " элементов. Локальный результат: " << local_result << std::endl;
    }

    // Завершаем работу MPI
    MPI_Finalize();

    return 0;
}