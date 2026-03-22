#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include "fft.c"
#define mysin my_sin
#define mycos my_cos
#define myexp my_exp
#define EPSILON 0.001
#define PI 3.14159265358979323846


void test_1d_fft() {
    printf("\n===========================================\n");
    printf("TEST 2: 1D FFT Validation\n");
    printf("===========================================\n");
    
    int n = 8;  // Test size
    float* x_real_iter = (float*)malloc(n * sizeof(float));
    float* x_imag_iter = (float*)malloc(n * sizeof(float));
    float* x_real_rec = (float*)malloc(n * sizeof(float));
    float* x_imag_rec = (float*)malloc(n * sizeof(float));
    float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

    generate_twiddle_factors(twiddle_real, twiddle_imag, n);
    
    // Test 2a: Impulse response (delta function)
    printf("\nTest 2a: Impulse Response\n");
    printf("Input: [1, 0, 0, 0, 0, 0, 0, 0]\n");
    
    // Initialize impulse for both versions
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (i == 0) ? 1.0 : 0.0;
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    // Run iterative FFT
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    
    // Run recursive FFT
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    // Check results (impulse should give all 1+0i)
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
    
    // Test 2b: DC signal (all ones)
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

    // Test 2c: Compare iterative vs recursive output
    printf("\n\nTest 2c: Iterative vs Recursive Consistency\n");
    printf("Testing if both methods produce same results...\n");
    
    // Create a test signal
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (float)(i % 3);  // Arbitrary test signal
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
}

// ============================================
// Function 2: Benchmark 1D FFT
// ============================================

void benchmark_1d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 2D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\tIterative(s)\tRecursive(s)\tSpeedup\n");
    printf("----\t------------\t------------\t-------\n");
    
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_sizes = 8;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        // Allocate arrays
        float* x_real_iter = (float*)malloc(n * sizeof(float));
        float* x_imag_iter = (float*)malloc(n * sizeof(float));
        float* x_real_rec = (float*)malloc(n * sizeof(float));
        float* x_imag_rec = (float*)malloc(n * sizeof(float));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

        generate_twiddle_factors(twiddle_real, twiddle_imag, n);
        
        // Initialize with test data
        for (int i = 0; i < n; i++) {
            x_real_iter[i] = (float)i;
            x_imag_iter[i] = 0.0;
            x_real_rec[i] = x_real_iter[i];
            x_imag_rec[i] = x_imag_iter[i];
        }
        
        // Benchmark iterative
        clock_t start = clock();
        fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
        clock_t end = clock();
        double time_iter = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Benchmark recursive
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
    }
    
    printf("\n");
}

// ============================================
// Function 3: Test Trigonometric Functions
// ============================================

void test_math_functions() {
    printf("\n===========================================\n");
    printf("TEST 1: Math Functions Validation\n");
    printf("===========================================\n");
    
    // Test points for trig functions
    float test_angles[] = {
        0.0,
        PI / 6.0,      // 30 degrees
        PI / 4.0,      // 45 degrees
        PI / 3.0,      // 60 degrees
        PI / 2.0,      // 90 degrees
        PI,            // 180 degrees
        3.0 * PI / 2.0,  // 270 degrees
        2.0 * PI,      // 360 degrees
        -PI / 4.0,     // -45 degrees
        -PI / 2.0      // -90 degrees
    };
    
    char* angle_names[] = {
        "0pi",
        "30pi",
        "45pi",
        "60pi",
        "90pi",
        "180pi",
        "270pi",
        "360pi",
        "-45pi",
        "-90pi"
    };
    
    // Test points for exponential function
    float test_exponents[] = {
        0.0,
        0.5,
        1.0,
        2.0,
        -1.0,
        -2.0,
        5.0,
        -5.0,
        0.1,
        -0.1
    };
    
    char* exp_names[] = {
        "0.0",
        "0.5",
        "1.0",
        "2.0",
        "-1.0",
        "-2.0",
        "5.0",
        "-5.0",
        "0.1",
        "-0.1"
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

// ============================================
// Function 4: Test 2D FFT
// ============================================

void test_2d_fft() {
    printf("\n===========================================\n");
    printf("TEST 3: 2D FFT Validation\n");
    printf("===========================================\n");
    
    int rows = 4;
    int cols = 4;
    int size = rows * cols;
    
    float** matrix_real = (float**)malloc(rows * sizeof(float*));
    float** matrix_imag = (float**)malloc(rows * sizeof(float*));
    float* twiddle_real = (float*)malloc(rows/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(rows/2 * sizeof(float));

    // Test 3a: 2D Impulse (delta at [0,0])
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
    
    // Test 3b: 2D DC signal (all ones)
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
                // DC component should be 16
                if (fabs(matrix_real[i][j] - 16.0) > EPSILON || 
                    fabs(matrix_imag[i][j]) > EPSILON) {
                    test_passed = 0;
                }
            } else {
                // All others should be ~0
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

// ============================================
// Function 5: Benchmark 2D FFT
// ============================================

void benchmark_2d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 2D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\t\tTime(s)\t\tOperations\n");
    printf("----\t\t-------\t\t----------\n");
    
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = 6;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        int total_size = n * n;
        
        float** matrix_real = (float**)malloc(n * sizeof(float*));
        float** matrix_imag = (float**)malloc(n * sizeof(float*));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float)); 
        
        // Initialize with test data
        for (int i = 0; i < n; i++) {
            matrix_real[i] = (float*)malloc(n * sizeof(float));
            matrix_imag[i] = (float*)malloc(n * sizeof(float));
            for (int j = 0; j < n; j++) {
                matrix_real[i][j] = (float)((i * n + j) % 10);
                matrix_imag[i][j] = 0.0;
            }
        }
        
        // Benchmark
        clock_t start = clock();
        fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, n, n, false, false);
        clock_t end = clock();
        
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Theoretical operations: 2 * n * n * log2(n)
        // (n row FFTs of size n, n column FFTs of size n)
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

// ============================================
// Function to Run All Tests
// ============================================

void run_all_tests() {
    printf("===========================================\n");
    printf("FFT TEST SUITE - MILESTONE 1\n");
    printf("===========================================\n");
    
    // Run all tests
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