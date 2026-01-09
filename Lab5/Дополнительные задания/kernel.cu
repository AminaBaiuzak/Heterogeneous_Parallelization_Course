#include <iostream>      // для ввода/вывода
#include <cuda.h>        // для работы с CUDA
#include <chrono>        // для измерения времени выполнения

using namespace std;

// ===========================================
// Параллельный MPMC стек на GPU
// ===========================================
struct StackGPU {
    int* data;    // указатель на массив данных на GPU
    int* top;     // указатель на переменную верхушки стека в глобальной памяти
    int capacity; // ёмкость стека

    // Инициализация структуры
    void init(int* buffer, int* top_ptr, int size) {
        data = buffer;      // массив данных
        top = top_ptr;      // указатель на верхушку
        capacity = size;    // максимальный размер
    }

    // Добавление элемента в стек (параллельно, атомарно)
    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1); // атомарное увеличение top
        if (pos < capacity) {        // проверяем, не вышли ли за предел
            data[pos] = value;       // записываем значение
            return true;
        }
        return false;
    }

    // Удаление элемента из стека (параллельно, атомарно)
    __device__ bool pop(int* value) {
        int pos = atomicSub(top, 1); // атомарное уменьшение top
        if (pos >= 0) {              // проверка, что элементы есть
            *value = data[pos];      // забираем значение
            return true;
        }
        return false;
    }
};

// ===========================================
// Параллельная MPMC очередь на GPU
// ===========================================
struct QueueGPU {
    int* data;       // массив данных на GPU
    int* head;       // указатель на начало очереди
    int* tail;       // указатель на конец очереди
    int capacity;    // максимальная ёмкость

    // Инициализация
    void init(int* buffer, int* head_ptr, int* tail_ptr, int size) {
        data = buffer;
        head = head_ptr;
        tail = tail_ptr;
        capacity = size;
    }

    // Добавление элемента в очередь (атомарно)
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1); // атомарно увеличиваем tail
        if (pos < capacity) {
            data[pos] = value;         // записываем элемент
            return true;
        }
        return false;
    }

    // Удаление элемента из очереди (атомарно)
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(head, 1); // атомарно увеличиваем head
        if (pos < *tail) {            // проверка, что очередь не пуста
            *value = data[pos];       // забираем значение
            return true;
        }
        return false;
    }
};

// ===========================================
// CUDA ядро для тестирования стека
// ===========================================
__global__ void test_stack_kernel(StackGPU stack, int num_ops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // уникальный индекс потока
    for (int i = 0; i < num_ops; i++) {             // каждый поток выполняет num_ops операций
        stack.push(tid + i);                        // push в стек
        int val;
        stack.pop(&val);                             // pop из стека
    }
}

// ===========================================
// CUDA ядро для тестирования очереди
// ===========================================
__global__ void test_queue_kernel(QueueGPU queue, int num_ops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < num_ops; i++) {
        queue.enqueue(tid + i); // enqueue
        int val;
        queue.dequeue(&val);    // dequeue
    }
}

// ===========================================
// Последовательный стек на CPU
// ===========================================
struct StackCPU {
    int* data;
    int top;
    int capacity;

    void init(int size) {
        data = new int[size]; // выделяем массив
        top = -1;             // пустой стек
        capacity = size;      // максимальный размер
    }

    bool push(int value) {
        if (top + 1 < capacity) {
            data[++top] = value; // увеличиваем top и записываем
            return true;
        }
        return false;
    }

    bool pop(int* value) {
        if (top >= 0) {
            *value = data[top--]; // забираем значение и уменьшаем top
            return true;
        }
        return false;
    }

    ~StackCPU() { delete[] data; } // освобождаем память
};

// ===========================================
// Последовательная очередь на CPU
// ===========================================
struct QueueCPU {
    int* data;
    int head, tail;
    int capacity;

    void init(int size) {
        data = new int[size];
        head = 0; tail = 0;
        capacity = size;
    }

    bool enqueue(int value) {
        if (tail < capacity) {
            data[tail++] = value; // записываем и увеличиваем tail
            return true;
        }
        return false;
    }

    bool dequeue(int* value) {
        if (head < tail) {
            *value = data[head++]; // забираем и увеличиваем head
            return true;
        }
        return false;
    }

    ~QueueCPU() { delete[] data; } // освобождаем память
};

// ===========================================
// Главная функция
// ===========================================
int main() {
    int N = 100000;               // размер структуры
    int threadsPerBlock = 256;    // потоков на блок
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // количество блоков
    int num_ops = 10;             // операций на поток

    // -------------------------
    // Выделяем память на GPU для стека
    // -------------------------
    int* d_stack_data, * d_stack_top;
    cudaMalloc(&d_stack_data, N * sizeof(int));
    cudaMalloc(&d_stack_top, sizeof(int));
    cudaMemset(d_stack_top, -1, sizeof(int)); // стек пустой

    // -------------------------
    // Выделяем память на GPU для очереди
    // -------------------------
    int* d_queue_data, * d_queue_head, * d_queue_tail;
    cudaMalloc(&d_queue_data, N * sizeof(int));
    cudaMalloc(&d_queue_head, sizeof(int));
    cudaMalloc(&d_queue_tail, sizeof(int));
    cudaMemset(d_queue_head, 0, sizeof(int));
    cudaMemset(d_queue_tail, 0, sizeof(int));

    // Инициализация структур
    StackGPU stack;
    stack.init(d_stack_data, d_stack_top, N);

    QueueGPU queue;
    queue.init(d_queue_data, d_queue_head, d_queue_tail, N);

    // -------------------------
    // Измеряем GPU стек
    // -------------------------
    auto start_gpu_stack = chrono::high_resolution_clock::now();
    test_stack_kernel << <blocks, threadsPerBlock >> > (stack, num_ops);
    cudaDeviceSynchronize(); // ждём завершения всех потоков
    auto end_gpu_stack = chrono::high_resolution_clock::now();
    cout << "Время Stack на GPU: "
        << chrono::duration<double, milli>(end_gpu_stack - start_gpu_stack).count()
        << " мс\n";

    // -------------------------
    // Измеряем GPU очередь
    // -------------------------
    auto start_gpu_queue = chrono::high_resolution_clock::now();
    test_queue_kernel << <blocks, threadsPerBlock >> > (queue, num_ops);
    cudaDeviceSynchronize();
    auto end_gpu_queue = chrono::high_resolution_clock::now();
    cout << "Время Queue на GPU: "
        << chrono::duration<double, milli>(end_gpu_queue - start_gpu_queue).count()
        << " мс\n";

    // -------------------------
    // Последовательный CPU стек
    // -------------------------
    StackCPU stack_cpu;
    stack_cpu.init(N);
    auto start_cpu_stack = chrono::high_resolution_clock::now();
    for (int t = 0; t < blocks * threadsPerBlock; t++) {
        for (int i = 0; i < num_ops; i++) {
            stack_cpu.push(t + i);
            int val;
            stack_cpu.pop(&val);
        }
    }
    auto end_cpu_stack = chrono::high_resolution_clock::now();
    cout << "Время Stack на CPU: "
        << chrono::duration<double, milli>(end_cpu_stack - start_cpu_stack).count()
        << " мс\n";

    // -------------------------
    // Последовательная CPU очередь
    // -------------------------
    QueueCPU queue_cpu;
    queue_cpu.init(N);
    auto start_cpu_queue = chrono::high_resolution_clock::now();
    for (int t = 0; t < blocks * threadsPerBlock; t++) {
        for (int i = 0; i < num_ops; i++) {
            queue_cpu.enqueue(t + i);
            int val;
            queue_cpu.dequeue(&val);
        }
    }
    auto end_cpu_queue = chrono::high_resolution_clock::now();
    cout << "Время Queue на CPU: "
        << chrono::duration<double, milli>(end_cpu_queue - start_cpu_queue).count()
        << " мс\n";

    // -------------------------
    // Освобождаем память GPU
    // -------------------------
    cudaFree(d_stack_data);
    cudaFree(d_stack_top);
    cudaFree(d_queue_data);
    cudaFree(d_queue_head);
    cudaFree(d_queue_tail);

    return 0;
}
