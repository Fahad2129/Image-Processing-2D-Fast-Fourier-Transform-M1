#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "fft.c"
#define mysin my_sin
#define mycos my_cos
#define myexp my_exp
#define PI 3.14159265358979323846
#define EPSILON 0.001

// Forward declarations of functions
static bool is_power_of_two(int n);
float my_sin(float x);
float my_cos(float x);
float my_exp(float x);
int log2_int(int n);
float mysin(float x);
float mycos(float x);
float myexp(float x);
void generate_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n);
void generate_inverse_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n);
unsigned int reverse_bits(unsigned int x, int log_n);
void bit_reverse_array(float *x_real, float *x_imag, int n);
void butterfly_iterative(float *x_real, float *x_imag, float *twiddle_real, float *twiddle_imag, int n);
void fft_1d_iterative(float *x_real, float *x_imag, float *twiddle_real, float *twiddle_imag, int n);
void butterfly_recursive(float *x_real, float *x_imag, float *twiddle_real, float *twiddle_imag, int n);
void fft_1d_recursive(float *x_real, float *x_imag, float *twiddle_real, float *twiddle_imag, int n);
void transpose(float **matrix, int rows, int cols);
void fft_2d(float **x_real, float **x_imag, float *twiddle_real, float *twiddle_imag, int rows, int cols, bool test, bool inverse);


// fft implementaion
static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

float my_sin(float x) {
    while (x > PI) x -= 2 * PI;
    while (x < -PI) x += 2 * PI;
    
    float result = 0.0;
    float term = x;
    float x_squared = x * x;
    
    for (int n = 1; n <= 15; n += 2) {
        result += term;
        term *= -x_squared / ((n + 1) * (n + 2));
    }
    
    return result;
}

float my_cos(float x) {
    while (x > PI) x -= 2 * PI;
    while (x < -PI) x += 2 * PI;
    
    float result = 0.0;
    float term = 1.0;
    float x_squared = x * x;
    
    for (int n = 0; n <= 14; n += 2) {
        result += term;
        term *= -x_squared / ((n + 1) * (n + 2));
    }
    
    return result;
}

float my_exp(float x) {
    if (x > 10.0) {
        float half = my_exp(x / 2.0);
        return half * half;
    }
    if (x < -10.0) {
        return 1.0 / my_exp(-x);
    }
    
    float result = 1.0;
    float term = 1.0;
    
    for (int n = 1; n <= 20; n++) {
        term *= x / n;
        result += term;
    }
    
    return result;
}

int log2_int(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

float mysin(float x) { return my_sin(x); }
float mycos(float x) { return my_cos(x); }
float myexp(float x) { return my_exp(x); }

void generate_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n) {
    if (n <= 1) {
        return;
    }
    
    for (int k = 0; k < n/2; k++) {
        float angle = -2.0 * PI * k / n;
        twiddle_real[k] = my_cos(angle);
        twiddle_imag[k] = my_sin(angle);
    }
}

void generate_inverse_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n) {
    if (n <= 1) {
        return;
    }
    
    for (int k = 0; k < n/2; k++) {
        float angle = 2.0 * PI * k / n;
        twiddle_real[k] = my_cos(angle);
        twiddle_imag[k] = my_sin(angle);
    }
}

unsigned int reverse_bits(unsigned int x, int log_n) {
    unsigned int result = 0;
    for (int i = 0; i < log_n; i++) {
        result <<= 1;
        result |= (x & 1);
        x >>= 1;
    }
    return result;
}

void bit_reverse_array(float *x_real, float *x_imag, int n) {
    int log_n = log2_int(n);
    
    for (unsigned int i = 0; i < n; i++) {
        unsigned int j = reverse_bits(i, log_n);
        
        if (i < j) {
            float temp = x_real[i];
            x_real[i] = x_real[j];
            x_real[j] = temp;
            
            temp = x_imag[i];
            x_imag[i] = x_imag[j];
            x_imag[j] = temp;
        }
    }
}

void butterfly_iterative(float *x_real, float *x_imag, float *twiddle_real, 
                         float *twiddle_imag, int n) {
    int log_n = log2_int(n);
    
    for (int stage = 1; stage <= log_n; stage++) {
        int m = 1 << stage;
        int m_half = m >> 1;
        
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m_half; j++) {
                int twiddle_index = (j * n) / m;
                
                float tw_real = twiddle_real[twiddle_index];
                float tw_imag = twiddle_imag[twiddle_index];
                
                int idx1 = k + j;
                int idx2 = k + j + m_half;
                
                float temp_real = tw_real * x_real[idx2] - tw_imag * x_imag[idx2];
                float temp_imag = tw_real * x_imag[idx2] + tw_imag * x_real[idx2];
                
                x_real[idx2] = x_real[idx1] - temp_real;
                x_imag[idx2] = x_imag[idx1] - temp_imag;
                x_real[idx1] = x_real[idx1] + temp_real;
                x_imag[idx1] = x_imag[idx1] + temp_imag;
            }
        }
    }
}

void fft_1d_iterative(float *x_real, float *x_imag, float *twiddle_real, 
                      float *twiddle_imag, int n) {
    if (n <= 1) {
        return;
    }
    if (!is_power_of_two(n)) {
        printf("Error: FFT length must be a power of 2 (got %d)\n", n);
        return;
    }
    bit_reverse_array(x_real, x_imag, n);
    butterfly_iterative(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

static void butterfly_recursive_impl(float *x_real, float *x_imag, float *twiddle_real,
                                     float *twiddle_imag, int n, int total_n) {
    if (n <= 1) {
        return;
    }

    int half = n / 2;
    butterfly_recursive_impl(x_real, x_imag, twiddle_real, twiddle_imag, half, total_n);
    butterfly_recursive_impl(x_real + half, x_imag + half, twiddle_real, twiddle_imag, half, total_n);

    for (int j = 0; j < half; j++) {
        int twiddle_index = (j * total_n) / n;
        float tw_real = twiddle_real[twiddle_index];
        float tw_imag = twiddle_imag[twiddle_index];

        float even_real = x_real[j];
        float even_imag = x_imag[j];
        float odd_real = x_real[j + half];
        float odd_imag = x_imag[j + half];

        float temp_real = tw_real * odd_real - tw_imag * odd_imag;
        float temp_imag = tw_real * odd_imag + tw_imag * odd_real;

        x_real[j] = even_real + temp_real;
        x_imag[j] = even_imag + temp_imag;
        x_real[j + half] = even_real - temp_real;
        x_imag[j + half] = even_imag - temp_imag;
    }
}

void butterfly_recursive(float *x_real, float *x_imag, float *twiddle_real,
                        float *twiddle_imag, int n) {
    butterfly_recursive_impl(x_real, x_imag, twiddle_real, twiddle_imag, n, n);
}

void fft_1d_recursive(float *x_real, float *x_imag, float *twiddle_real,
                     float *twiddle_imag, int n) {
    if (n <= 1) {
        return;
    }
    if (!is_power_of_two(n)) {
        printf("Error: FFT length must be a power of 2 (got %d)\n", n);
        return;
    }
    bit_reverse_array(x_real, x_imag, n);
    butterfly_recursive(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

void transpose(float **matrix, int rows, int cols) {
    if (rows != cols) {
        return;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < cols; j++) {
            float temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

void fft_2d(float **x_real, float **x_imag, float *twiddle_real,
           float *twiddle_imag, int rows, int cols, bool test, bool inverse) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if (!is_power_of_two(rows) || !is_power_of_two(cols)) {
        printf("Error: 2D FFT dimensions must both be powers of 2 (got %d x %d)\n", rows, cols);
        return;
    }
    
    if (inverse) {
        generate_inverse_twiddle_factors(twiddle_real, twiddle_imag, cols);
    } else {
        generate_twiddle_factors(twiddle_real, twiddle_imag, cols);
    }
    
    for (int i = 0; i < rows; i++) {
        fft_1d_iterative(x_real[i], x_imag[i], twiddle_real, twiddle_imag, cols);
    }
    
    if (test) {
        printf("\nAfter row FFTs:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("(%.2f+%.2fi) ", x_real[i][j], x_imag[i][j]);
            }
            printf("\n");
        }
    }
    
    if (rows == cols) {
        transpose(x_real, rows, cols);
        transpose(x_imag, rows, cols);

        if (inverse) {
            generate_inverse_twiddle_factors(twiddle_real, twiddle_imag, rows);
        } else {
            generate_twiddle_factors(twiddle_real, twiddle_imag, rows);
        }

        for (int i = 0; i < cols; i++) {
            fft_1d_iterative(x_real[i], x_imag[i], twiddle_real, twiddle_imag, rows);
        }

        transpose(x_real, cols, rows);
        transpose(x_imag, cols, rows);
    } else {
        float *col_real = (float *)malloc(rows * sizeof(float));
        float *col_imag = (float *)malloc(rows * sizeof(float));

        if (col_real == NULL || col_imag == NULL) {
            printf("Error: failed to allocate temporary column buffers for 2D FFT\n");
            free(col_real);
            free(col_imag);
            return;
        }

        if (inverse) {
            generate_inverse_twiddle_factors(twiddle_real, twiddle_imag, rows);
        } else {
            generate_twiddle_factors(twiddle_real, twiddle_imag, rows);
        }

        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                col_real[r] = x_real[r][c];
                col_imag[r] = x_imag[r][c];
            }

            fft_1d_iterative(col_real, col_imag, twiddle_real, twiddle_imag, rows);

            for (int r = 0; r < rows; r++) {
                x_real[r][c] = col_real[r];
                x_imag[r][c] = col_imag[r];
            }
        }

        free(col_real);
        free(col_imag);
    }
}

// TEST SUITE

void test_1d_fft() {
    printf("\n===========================================\n");
    printf("TEST 2: 1D FFT Validation\n");
    printf("===========================================\n");
    
    int n = 8;
    float* x_real_iter = (float*)malloc(n * sizeof(float));
    float* x_imag_iter = (float*)malloc(n * sizeof(float));
    float* x_real_rec = (float*)malloc(n * sizeof(float));
    float* x_imag_rec = (float*)malloc(n * sizeof(float));
    float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

    generate_twiddle_factors(twiddle_real, twiddle_imag, n);
    
    // Test 2a: Impulse response
    printf("\nTest 2a: Impulse Response\n");
    printf("Input: [1, 0, 0, 0, 0, 0, 0, 0]\n");
    
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (i == 0) ? 1.0 : 0.0;
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    printf("Expected: All bins should be 1.0 + 0.0i\n");
    printf("\nIterative FFT Results:\n");
    int iter_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_iter[i], x_imag_iter[i]);
        if (fabs(x_real_iter[i] - 1.0) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
            printf(" [FAIL]");
            iter_correct = 0;
        } else {
            printf(" [PASS]");
        }
        printf("\n");
    }
    
    printf("\nRecursive FFT Results:\n");
    int rec_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_rec[i], x_imag_rec[i]);
        if (fabs(x_real_rec[i] - 1.0) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
            printf(" [FAIL]");
            rec_correct = 0;
        } else {
            printf(" [PASS]");
        }
        printf("\n");
    }
    
    printf("\nTest 2a Result: Iterative %s, Recursive %s\n", 
           iter_correct ? "PASS" : "FAIL",
           rec_correct ? "PASS" : "FAIL");
    
    // Test 2b: DC signal
    printf("\n\nTest 2b: DC Signal\n");
    printf("Input: [1, 1, 1, 1, 1, 1, 1, 1]\n");
    
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = 1.0;
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    printf("Expected: Bin[0] = 8.0, all others = 0.0\n");
    printf("\nIterative FFT Results:\n");
    iter_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_iter[i], x_imag_iter[i]);
        if (i == 0) {
            if (fabs(x_real_iter[i] - 8.0) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
                printf(" [FAIL]");
                iter_correct = 0;
            } else {
                printf(" [PASS]");
            }
        } else {
            if (fabs(x_real_iter[i]) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
                printf(" [FAIL]");
                iter_correct = 0;
            } else {
                printf(" [PASS]");
            }
        }
        printf("\n");
    }
    
    printf("\nRecursive FFT Results:\n");
    rec_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_rec[i], x_imag_rec[i]);
        if (i == 0) {
            if (fabs(x_real_rec[i] - 8.0) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
                printf(" [FAIL]");
                rec_correct = 0;
            } else {
                printf(" [PASS]");
            }
        } else {
            if (fabs(x_real_rec[i]) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
                printf(" [FAIL]");
                rec_correct = 0;
            } else {
                printf(" [PASS]");
            }
        }
        printf("\n");
    }
    
    printf("\nTest 2b Result: Iterative %s, Recursive %s\n", 
           iter_correct ? "PASS" : "FAIL",
           rec_correct ? "PASS" : "FAIL");

    // Test 2c: Compare iterative vs recursive
    printf("\n\nTest 2c: Iterative vs Recursive Consistency\n");
    printf("Testing if both methods produce same results...\n");
    
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (float)(i % 3);
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    int match = 1;
    for (int i = 0; i < n; i++) {
        float diff_real = fabs(x_real_iter[i] - x_real_rec[i]);
        float diff_imag = fabs(x_imag_iter[i] - x_imag_rec[i]);
        
        if (diff_real > EPSILON || diff_imag > EPSILON) {
            printf("  Bin[%d] MISMATCH: Iter=(%.3f+%.3fi), Rec=(%.3f+%.3fi)\n",
                   i, x_real_iter[i], x_imag_iter[i], 
                   x_real_rec[i], x_imag_rec[i]);
            match = 0;
        }
    }
    
    if (match) {
        printf("  All bins match! [PASS]\n");
    } else {
        printf("  Some bins don't match [FAIL]\n");
    }
    
    printf("\nTest 2c Result: %s\n", match ? "PASS" : "FAIL");
    
    free(x_real_iter);
    free(x_imag_iter);
    free(x_real_rec);
    free(x_imag_rec);
    free(twiddle_real);
    free(twiddle_imag);
}

void benchmark_1d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 1D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\tIterative(s)\tRecursive(s)\tSpeedup\n");
    printf("----\t------------\t------------\t-------\n");
    
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_sizes = 8;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        float* x_real_iter = (float*)malloc(n * sizeof(float));
        float* x_imag_iter = (float*)malloc(n * sizeof(float));
        float* x_real_rec = (float*)malloc(n * sizeof(float));
        float* x_imag_rec = (float*)malloc(n * sizeof(float));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

        generate_twiddle_factors(twiddle_real, twiddle_imag, n);
        
        for (int i = 0; i < n; i++) {
            x_real_iter[i] = (float)i;
            x_imag_iter[i] = 0.0;
            x_real_rec[i] = x_real_iter[i];
            x_imag_rec[i] = x_imag_iter[i];
        }
        
        clock_t start = clock();
        fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
        clock_t end = clock();
        double time_iter = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        start = clock();
        fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
        end = clock();
        double time_rec = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        double speedup = time_rec / time_iter;
        
        printf("%d\t%.6f\t%.6f\t%.2fx\n", n, time_iter, time_rec, speedup);
        
        free(x_real_iter);
        free(x_imag_iter);
        free(x_real_rec);
        free(x_imag_rec);
        free(twiddle_real);
        free(twiddle_imag);
    }
    
    printf("\n");
}

void test_math_functions() {
    printf("\n===========================================\n");
    printf("TEST 1: Math Functions Validation\n");
    printf("===========================================\n");
    
    float test_angles[] = {
        0.0,
        PI / 6.0,
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        PI,
        3.0 * PI / 2.0,
        2.0 * PI,
        -PI / 4.0,
        -PI / 2.0
    };
    
    char* angle_names[] = {
        "0pi", "30pi", "45pi", "60pi", "90pi",
        "180pi", "270pi", "360pi", "-45pi", "-90pi"
    };
    
    float test_exponents[] = {
        0.0, 0.5, 1.0, 2.0, -1.0,
        -2.0, 5.0, -5.0, 0.1, -0.1
    };
    
    char* exp_names[] = {
        "0.0", "0.5", "1.0", "2.0", "-1.0",
        "-2.0", "5.0", "-5.0", "0.1", "-0.1"
    };
    
    int num_trig_tests = 10;
    int num_exp_tests = 10;
    int sin_passed = 0;
    int cos_passed = 0;
    int exp_passed = 0;
    
    printf("\nTesting mysin():\n");
    printf("Angle\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_trig_tests; i++) {
        float your_sin = mysin(test_angles[i]);
        float expected_sin = sin(test_angles[i]);
        float error = fabs(your_sin - expected_sin);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               angle_names[i], your_sin, expected_sin, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            sin_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\nTesting mycos():\n");
    printf("Angle\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_trig_tests; i++) {
        float your_cos = mycos(test_angles[i]);
        float expected_cos = cos(test_angles[i]);
        float error = fabs(your_cos - expected_cos);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               angle_names[i], your_cos, expected_cos, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            cos_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\nTesting myexp():\n");
    printf("x\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_exp_tests; i++) {
        float your_exp = myexp(test_exponents[i]);
        float expected_exp = exp(test_exponents[i]);
        float error = fabs(your_exp - expected_exp);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               exp_names[i], your_exp, expected_exp, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            exp_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\n===========================================\n");
    printf("Summary: sin() %d/%d passed, cos() %d/%d passed, exp() %d/%d passed\n", 
           sin_passed, num_trig_tests, cos_passed, num_trig_tests, exp_passed, num_exp_tests);
    printf("===========================================\n");
}

void test_2d_fft() {
    printf("\n===========================================\n");
    printf("TEST 3: 2D FFT Validation\n");
    printf("===========================================\n");
    
    int rows = 4;
    int cols = 4;
    
    float** matrix_real = (float**)malloc(rows * sizeof(float*));
    float** matrix_imag = (float**)malloc(rows * sizeof(float*));
    float* twiddle_real = (float*)malloc(rows/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(rows/2 * sizeof(float));

    // Test 3a: 2D Impulse
    printf("\nTest 3a: 2D Impulse Response\n");
    printf("Input: Delta function at position [0,0]\n");
    
    for (int i = 0; i < rows; i++) {
        matrix_real[i] = (float*)malloc(cols * sizeof(float));
        matrix_imag[i] = (float*)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            if (i == 0 && j == 0) {
                matrix_real[i][j] = 1.0;
                matrix_imag[i][j] = 0.0;
            } else {
                matrix_real[i][j] = 0.0;
                matrix_imag[i][j] = 0.0;
            }
        }
    }
    
    printf("Input matrix (real part):\n");
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix_real[i][j]);
        }
        printf("\n");
    }
    
    fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, rows, cols, true, false);
    
    printf("\nExpected: All frequency bins should be 1.0 + 0.0i\n");
    printf("Output (real + imag):\n");
    
    int test_passed = 1;
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("(%.2f+%.2fi) ", matrix_real[i][j], matrix_imag[i][j]);
            
            if (fabs(matrix_real[i][j] - 1.0) > EPSILON || 
                fabs(matrix_imag[i][j]) > EPSILON) {
                test_passed = 0;
            }
        }
        printf("\n");
    }
    
    printf("\nTest 3a Result: %s\n", test_passed ? "PASS" : "FAIL");
    
    // Test 3b: 2D DC signal
    printf("\n\nTest 3b: 2D DC Signal\n");
    printf("Input: All ones\n");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_real[i][j] = 1.0;
            matrix_imag[i][j] = 0.0;
        }
    }
    
    printf("Input matrix (real part):\n");
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix_real[i][j]);
        }
        printf("\n");
    }
    
    fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, rows, cols, true, false);
    
    printf("\nExpected: DC bin [0,0] = 16.0, all others = 0.0\n");
    printf("Output (real + imag):\n");
    
    test_passed = 1;
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("(%.2f+%.2fi) ", matrix_real[i][j], matrix_imag[i][j]);
            
            if (i == 0 && j == 0) {
                if (fabs(matrix_real[i][j] - 16.0) > EPSILON || 
                    fabs(matrix_imag[i][j]) > EPSILON) {
                    test_passed = 0;
                }
            } else {
                if (fabs(matrix_real[i][j]) > EPSILON || 
                    fabs(matrix_imag[i][j]) > EPSILON) {
                    test_passed = 0;
                }
            }
        }
        printf("\n");
    }
    
    printf("\nTest 3b Result: %s\n", test_passed ? "PASS" : "FAIL");
    
    free(twiddle_real);
    free(twiddle_imag);

    for (int i = 0; i < rows; i++) {
        free(matrix_real[i]);
        free(matrix_imag[i]);
    }
    free(matrix_real);
    free(matrix_imag);
}

void benchmark_2d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 2D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\t\tTime(s)\t\tOperations\n");
    printf("----\t\t-------\t\t----------\n");
    
    int sizes[] = {128, 256, 512, 1024};
    int num_sizes = 4;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        float** matrix_real = (float**)malloc(n * sizeof(float*));
        float** matrix_imag = (float**)malloc(n * sizeof(float*));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float)); 
        
        for (int i = 0; i < n; i++) {
            matrix_real[i] = (float*)malloc(n * sizeof(float));
            matrix_imag[i] = (float*)malloc(n * sizeof(float));
            for (int j = 0; j < n; j++) {
                matrix_real[i][j] = (float)((i * n + j) % 10);
                matrix_imag[i][j] = 0.0;
            }
        }
        
        clock_t start = clock();
        fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, n, n, false, false);
        clock_t end = clock();
        
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        int log_n = 0;
        int temp = n;
        while (temp > 1) {
            log_n++;
            temp >>= 1;
        }
        long operations = 2L * n * n * log_n;
        
        printf("%dx%d\t\t%.6f\t%ld\n", n, n, time_taken, operations);
        
        for (int i = 0; i < n; i++) {
            free(matrix_real[i]);
            free(matrix_imag[i]);
        }
        free(matrix_real);
        free(matrix_imag);
        free(twiddle_real);
        free(twiddle_imag);
    }
    
    printf("\n");
}

void run_all_tests() {
    printf("===========================================\n");
    printf("FFT TEST SUITE - MILESTONE 1\n");
    printf("===========================================\n");
    
    test_math_functions();
    test_1d_fft();
    benchmark_1d_fft();
    test_2d_fft();
    benchmark_2d_fft();
    
    printf("\n===========================================\n");
    printf("ALL TESTS COMPLETE\n");
    printf("===========================================\n");
}

int main() {
    run_all_tests();
    return 0;
}