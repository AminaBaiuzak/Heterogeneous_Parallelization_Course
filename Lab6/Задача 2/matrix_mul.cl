// Ядро для параллельного умножения матриц
__kernel void matrix_mul(
    __global const float* A,  // Матрица A (N x M)
    __global const float* B,  // Матрица B (M x K)
    __global float* C,        // Результат C (N x K)
    const int N,              // Строки A
    const int M,              // Столбцы A / строки B
    const int K               // Столбцы B
) {
    int row = get_global_id(0); // индекс строки
    int col = get_global_id(1); // индекс столбца

    if (row < N && col < K) {    // защита от лишних потоков
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
