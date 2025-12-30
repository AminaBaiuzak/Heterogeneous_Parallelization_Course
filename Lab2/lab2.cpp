#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

// Псевдоним для удобного использования часов высокого разрешения
using clk = std::chrono::high_resolution_clock;


// Функция заполнения массива случайными числами
// arr  — указатель на массив
// size — количество элементов массива
void fill_array(int* arr, int size) {

    // Проходим по всем элементам массива
    for (int i = 0; i < size; i++)
        // Записываем случайное число в диапазоне [0; 99999]
        arr[i] = std::rand() % 100000;
}

// Функция копирования одного массива в другой
// src — исходный массив
// dst — массив, в который копируем данные
// size — количество элементов
void copy_array(int* src, int* dst, int size) {

    // Копируем каждый элемент по индексу
    for (int i = 0; i < size; i++)
        dst[i] = src[i];
}

// -------------------------------------------------
// ЗАДАНИЕ 1
// ПОСЛЕДОВАТЕЛЬНЫЕ СОРТИРОВКИ
// -------------------------------------------------

// Сортировка пузырьком (последовательная)
void bubble_sort_seq(int* arr, int size) {

    // Внешний цикл — количество проходов по массиву
    // После каждого прохода самый большой элемент
    // перемещается в конец массива
    for (int i = 0; i < size - 1; i++) {

        // Внутренний цикл — проход по неотсортированной части массива
        // size - 1 - i, потому что последние i элементов уже отсортированы
        for (int j = 0; j < size - 1 - i; j++) {

            // Сравниваем два соседних элемента
            if (arr[j] > arr[j + 1]) {

                // Если порядок неверный — меняем элементы местами
                int temp = arr[j];       // временно сохраняем arr[j]
                arr[j] = arr[j + 1];     // переносим правый элемент влево
                arr[j + 1] = temp;       // возвращаем сохранённый элемент
            }
        }
    }
}

// Сортировка выбором (последовательная)
void selection_sort_seq(int* arr, int size) {

    // Проходим по массиву, выбирая позицию для минимального элемента
    for (int i = 0; i < size - 1; i++) {

        // Считаем, что минимальный элемент находится на позиции i
        int min_index = i;

        // Ищем минимальный элемент в оставшейся части массива
        for (int j = i + 1; j < size; j++) {
            if (arr[j] < arr[min_index])
                min_index = j;
        }

        // Меняем местами текущий элемент и найденный минимальный
        int temp = arr[i];
        arr[i] = arr[min_index];
        arr[min_index] = temp;
    }
}

// Сортировка вставками (последовательная)
void insertion_sort_seq(int* arr, int size) {

    // Начинаем со второго элемента, считая первый отсортированным
    for (int i = 1; i < size; i++) {

        // Текущий элемент, который нужно вставить
        int key = arr[i];

        // Индекс предыдущего элемента
        int j = i - 1;

        // Сдвигаем элементы вправо, пока не найдём место для key
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        // Вставляем элемент на правильную позицию
        arr[j + 1] = key;
    }
}

// -------------------------------------------------
// ЗАДАНИЕ 2
// ПАРАЛЛЕЛЬНЫЕ СОРТИРОВКИ (OpenMP)
// -------------------------------------------------

// Параллельная пузырьковая сортировка (odd-even)
void bubble_sort_parallel(int* arr, int size) {

    // Количество фаз равно размеру массива
    for (int phase = 0; phase < size; phase++) {

        // ЧЁТНАЯ ФАЗА:
        // сравниваем элементы (0,1), (2,3), (4,5) и т.д.
    #pragma omp parallel for
        for (int i = 0; i < size - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }

        // НЕЧЁТНАЯ ФАЗА:
        // сравниваем элементы (1,2), (3,4), (5,6) и т.д.
#pragma omp parallel for
        for (int i = 1; i < size - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

// Параллельная сортировка выбором
void selection_sort_parallel(int* arr, int size) {

    for (int i = 0; i < size - 1; i++) {

        // Глобальный индекс минимального элемента
        int min_index = i;

        // Параллельная область
#pragma omp parallel
        {
            // Локальный минимум для каждого потока
            int local_min = min_index;

            // Параллельный поиск минимума
#pragma omp for nowait
            for (int j = i + 1; j < size; j++) {
                if (arr[j] < arr[local_min])
                    local_min = j;
            }

            // Критическая секция для обновления общего минимума
#pragma omp critical
            {
                if (arr[local_min] < arr[min_index])
                    min_index = local_min;
            }
        }

        // Обмен элементов
        int temp = arr[i];
        arr[i] = arr[min_index];
        arr[min_index] = temp;
    }
}

// Параллельная сортировка вставками
// Практически не ускоряется из-за зависимостей между элементами
void insertion_sort_parallel(int* arr, int size) {

#pragma omp parallel for
    for (int i = 1; i < size; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}

// -------------------------------------------------
// ЗАДАНИЕ 3
// ИЗМЕРЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ
// -------------------------------------------------

// Универсальная функция для измерения времени сортировки
void test_sort(const char* name,
    void (*sort_func)(int*, int),
    int* arr,
    int size)
{
    // Запоминаем момент начала
    auto start = clk::now();

    // Вызываем функцию сортировки
    sort_func(arr, size);

    // Запоминаем момент окончания
    auto end = clk::now();

    // Вычисляем время выполнения в миллисекундах
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Вывод результата
    std::cout << name << " | Time: " << time_ms << " ms\n";
}


void check_omp_support() {
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

int main() {

    setlocale(LC_ALL, "Russian");

    check_omp_support();

    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Вывод максимального количества потоков OpenMP
    std::cout << "Максимальное доступное количество потоков OpenMP: "
        << omp_get_max_threads() << "\n\n";

    // Размеры массивов для тестирования
    int sizes[] = { 1000, 10000, 100000 };

    // Тестирование сортировок для каждого размера массива
    for (int size : sizes) {

        std::cout << "Размер массива: " << size << "\n";

        // Динамическое выделение памяти
        int* original = new int[size];
        int* arr = new int[size];

        // Заполнение исходного массива случайными числами
        fill_array(original, size);

        // ---------------- ПУЗЫРЬКОВАЯ СОРТИРОВКА ----------------

        copy_array(original, arr, size);
        test_sort("Пузырьковая сортировка (последовательная)",
            bubble_sort_seq, arr, size);

        copy_array(original, arr, size);
        test_sort("Пузырьковая сортировка (параллельная)",
            bubble_sort_parallel, arr, size);

        std::cout << "\n";

        // ---------------- СОРТИРОВКА ВЫБОРОМ ----------------

        copy_array(original, arr, size);
        test_sort("Сортировка выбором (последовательная)",
            selection_sort_seq, arr, size);

        copy_array(original, arr, size);
        test_sort("Сортировка выбором (параллельная)",
            selection_sort_parallel, arr, size);

        std::cout << "\n";

        // ---------------- СОРТИРОВКА ВСТАВКАМИ ----------------

        copy_array(original, arr, size);
        test_sort("Сортировка вставками (последовательная)",
            insertion_sort_seq, arr, size);

        copy_array(original, arr, size);
        test_sort("Сортировка вставками (параллельная)",
            insertion_sort_parallel, arr, size);

        std::cout << "\n";

        // Освобождение памяти
        delete[] original;
        delete[] arr;
    }

    return 0;
}
