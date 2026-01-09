#include <iostream>
#include <cuda.h>
#include <chrono>

using namespace std;

// ===========================================
// ПАРАЛЛЕЛЬНЫЙ СТЕК
// ===========================================
struct Stack {
    int* data;    // массив для элементов стека
    int* top;     // указатель на верхний элемент в глобальной памяти
    int capacity; // емкость стека

    // Хостовая инициализация структуры
    void init(int* buffer, int* top_ptr, int size) {
        data = buffer;
        top = top_ptr;
        capacity = size;
    }

    // Push выполняется на GPU
    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1); // атомарно увеличиваем top и получаем индекс
        if (pos < capacity) {
            data[pos] = value;       // записываем значение
            return true;
        }
        return false;
    }

    // Pop выполняется на GPU
    __device__ bool pop(int* value) {
        int pos = atomicSub(top, 1); // атомарно уменьшаем top
        if (pos >= 0) {
            *value = data[pos];      // получаем значение
            return true;
        }
        return false;
    }
};

// ===========================================
// ПАРАЛЛЕЛЬНАЯ ОЧЕРЕДЬ
// ===========================================
struct Queue {
    int* data;    // массив для элементов очереди
    int* head;    // индекс головы очереди
    int* tail;    // индекс хвоста очереди
    int capacity; // емкость очереди

    // Хостовая инициализация структуры
    void init(int* buffer, int* head_ptr, int* tail_ptr, int size) {
        data = buffer;
        head = head_ptr;
        tail = tail_ptr;
        capacity = size;
    }

    // Enqueue выполняется на GPU
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1); // атомарно увеличиваем tail
        if (pos < capacity) {
            data[pos] = value;       // записываем значение
            return true;
        }
        return false;
    }

    // Dequeue выполняется на GPU
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(head, 1); // атомарно увеличиваем head
        if (pos < *tail) {
            *value = data[pos];      // читаем значение
            return true;
        }
        return false;
    }
};

// ===========================================
// Ядро CUDA для тестирования стека
// Каждый поток пытается сделать push и pop
// ===========================================
__global__ void test_stack_kernel(Stack stack, int num_ops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < num_ops; i++) {
        stack.push(tid + i); // push значение
        int val;
        stack.pop(&val);     // сразу pop
    }
}

// ===========================================
// Ядро CUDA для тестирования очереди
// Каждый поток пытается сделать enqueue и dequeue
// ===========================================
__global__ void test_queue_kernel(Queue queue, int num_ops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < num_ops; i++) {
        queue.enqueue(tid + i);
        int val;
        queue.dequeue(&val);
    }
}

// ===========================================
// Главная функция
// ===========================================
int main() {

    int N = 100000;            // размер стека/очереди
    int threadsPerBlock = 256; // количество потоков в блоке
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int num_ops = 10;          // количество операций на поток

    // -------------------------
    // Выделяем память для стека
    // -------------------------
    int* d_stack_data;
    int* d_stack_top;
    cudaMalloc(&d_stack_data, N * sizeof(int));
    cudaMalloc(&d_stack_top, sizeof(int));
    cudaMemset(d_stack_top, -1, sizeof(int)); // стек пустой

    Stack stack;
    stack.init(d_stack_data, d_stack_top, N);

    // -------------------------
    // Выделяем память для очереди
    // -------------------------
    int* d_queue_data;
    int* d_queue_head;
    int* d_queue_tail;
    cudaMalloc(&d_queue_data, N * sizeof(int));
    cudaMalloc(&d_queue_head, sizeof(int));
    cudaMalloc(&d_queue_tail, sizeof(int));
    cudaMemset(d_queue_head, 0, sizeof(int));
    cudaMemset(d_queue_tail, 0, sizeof(int));

    Queue queue;
    queue.init(d_queue_data, d_queue_head, d_queue_tail, N);

    // -------------------------
    // Тестируем стек
    // -------------------------
    auto start_stack = chrono::high_resolution_clock::now();
    test_stack_kernel << <blocks, threadsPerBlock >> > (stack, num_ops);
    cudaDeviceSynchronize();
    auto end_stack = chrono::high_resolution_clock::now();

    cout << "Время операций push/pop стека: "
        << chrono::duration<double, milli>(end_stack - start_stack).count()
        << " мс\n";

    // -------------------------
    // Тестируем очередь
    // -------------------------
    auto start_queue = chrono::high_resolution_clock::now();
    test_queue_kernel << <blocks, threadsPerBlock >> > (queue, num_ops);
    cudaDeviceSynchronize();
    auto end_queue = chrono::high_resolution_clock::now();

    cout << "Время операций enqueue/dequeue очереди: "
        << chrono::duration<double, milli>(end_queue - start_queue).count()
        << " мс\n";

    // -------------------------
    // Освобождаем память
    // -------------------------
    cudaFree(d_stack_data);
    cudaFree(d_stack_top);
    cudaFree(d_queue_data);
    cudaFree(d_queue_head);
    cudaFree(d_queue_tail);

    return 0;
}
