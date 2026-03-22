#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

 // CONSTANTS
#define PI 3.14159265358979323846f

 // MATH FUNCTIONS no math library used

/* Reduce x into [-PI, PI] */
static float wrap_angle(float x) {
    while (x >  PI) x -= 2.0f * PI;
    while (x < -PI) x += 2.0f * PI;
    return x;
}

/* sin(x) via Taylor series - converges well for |x| <= PI */
float my_sin(float x) {
    x = wrap_angle(x);
    float x2  = x * x;
    /* 9-term Taylor: x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! */
    float result = x * (1.0f + x2 * (-1.0f/6.0f
                  + x2 * ( 1.0f/120.0f
                  + x2 * (-1.0f/5040.0f
                  + x2 * ( 1.0f/362880.0f
                  + x2 * (-1.0f/39916800.0f))))));
    return result;
}

/* cos(x) = sin(x + PI/2) */
float my_cos(float x) {
    return my_sin(x + PI / 2.0f);
}

/* exp(x) via range reduction + Taylor series
 * exp(x) = exp(n) * exp(r) where n = round(x/ln2), r = x - n*ln2
 * We precompute exp(n) using integer powers of e^ln2 = 2.
 * For |r| <= 0.5*ln2, Taylor is very accurate.
 */
float my_exp(float x) {
    /* Clamp to avoid overflow */
    if (x > 88.0f)  return 3.40282347e+38f;
    if (x < -88.0f) return 0.0f;

    /* Range reduction: x = n*ln2 + r */
    const float LN2 = 0.693147180559945f;
    int n = (int)(x / LN2 + 0.5f);
    float r = x - (float)n * LN2;

    /* Taylor for exp(r), |r| <= 0.35 */
    float r2 = r * r;
    float er = 1.0f + r * (1.0f
             + r * (0.5f
             + r * (1.0f/6.0f
             + r * (1.0f/24.0f
             + r * (1.0f/120.0f
             + r * (1.0f/720.0f
             + r * (1.0f/5040.0f)))))));

    /* Multiply by 2^n using ldexpf */
    return ldexpf(er, n);
}

/* Integer log2 */
int log2_int(int n) {
    int log = 0;
    while (n > 1) { n >>= 1; log++; }
    return log;
}

//  * TWIDDLE FACTOR GENERATION
//  * For DIT-FFT:  W_N^k = e^(-j*2*PI*k/N)  (forward)
//  *               W_N^k = e^(+j*2*PI*k/N)  (inverse)
//  * Array size = N/2
void generate_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n) {
    int half = n / 2;
    for (int k = 0; k < half; k++) {
        float angle = -2.0f * PI * (float)k / (float)n;
        twiddle_real[k] = my_cos(angle);
        twiddle_imag[k] = my_sin(angle);
    }
}

void generate_inverse_twiddle_factors(float *twiddle_real, float *twiddle_imag, int n) {
    int half = n / 2;
    for (int k = 0; k < half; k++) {
        float angle = 2.0f * PI * (float)k / (float)n;
        twiddle_real[k] = my_cos(angle);
        twiddle_imag[k] = my_sin(angle);
    }
}

 // BIT REVERSAL
unsigned int reverse_bits(unsigned int x, int log_n) {
    unsigned int result = 0;
    for (int i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void bit_reverse_array(float *x_real, float *x_imag, int n) {
    int log_n = log2_int(n);
    for (int i = 0; i < n; i++) {
        unsigned int j = reverse_bits((unsigned int)i, log_n);
        if (j > (unsigned int)i) {
            /* Swap real */
            float tmp   = x_real[i];
            x_real[i]   = x_real[j];
            x_real[j]   = tmp;
            /* Swap imag */
            tmp          = x_imag[i];
            x_imag[i]    = x_imag[j];
            x_imag[j]    = tmp;
        }
    }
}


 //1D FFT - ITERATIVE (Cooley-Tukey DIT)
void butterfly_iterative(float *x_real, float *x_imag,
                         float *twiddle_real, float *twiddle_imag, int n) {
    int log_n = log2_int(n);

    /* Stage s: butterfly span = 2^s */
    for (int s = 1; s <= log_n; s++) {
        int m     = 1 << s;        /* butterfly group size */
        int half  = m >> 1;        /* half group size      */
        int step  = n / m;         /* twiddle stride       */

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < half; j++) {
                int tw_idx = j * step;  /* W_n^(j * n/m) */

                float wr = twiddle_real[tw_idx];
                float wi = twiddle_imag[tw_idx];

                float ur = x_real[k + j];
                float ui = x_imag[k + j];
                float vr = x_real[k + j + half];
                float vi = x_imag[k + j + half];

                /* twiddle * v */
                float tr = wr * vr - wi * vi;
                float ti = wr * vi + wi * vr;

                x_real[k + j]        = ur + tr;
                x_imag[k + j]        = ui + ti;
                x_real[k + j + half] = ur - tr;
                x_imag[k + j + half] = ui - ti;
            }
        }
    }
}

void fft_1d_iterative(float *x_real, float *x_imag,
                      float *twiddle_real, float *twiddle_imag, int n) {
    bit_reverse_array(x_real, x_imag, n);
    butterfly_iterative(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

/* =========================================================
 * 1D FFT - RECURSIVE (Cooley-Tukey DIT)
 *
 * The twiddle arrays were generated for the FULL transform size (full_n),
 * so they contain entries W[0..full_n/2-1] where W[k] = e^(-j2πk/full_n).
 *
 * For a sub-problem of local size n, the required twiddle for butterfly
 * position k is W_n^k = e^(-j2πk/n) = W_full_n^(k * full_n/n).
 * So the stride into the twiddle array is:  step = full_n / n
 * and the index used is:                    twiddle[k * step]
 *
 * carry full_n through the recursion via the helper below.
 * ========================================================= */

/* Internal recursive helper - full_n is the original transform size */
static void butterfly_recursive_helper(float *x_real, float *x_imag,
                                        float *twiddle_real, float *twiddle_imag,
                                        int n, int full_n) {
    if (n <= 1) return;

    int half = n / 2;
    /* stride: twiddle was built for full_n, current sub-problem is n */
    int step = full_n / n;

    /* Recurse on lower half (even-indexed after bit-reversal) */
    butterfly_recursive_helper(x_real,        x_imag,        twiddle_real, twiddle_imag, half, full_n);
    /* Recurse on upper half (odd-indexed after bit-reversal) */
    butterfly_recursive_helper(x_real + half, x_imag + half, twiddle_real, twiddle_imag, half, full_n);

    /* Combine using twiddle factors from the pre-built array */
    for (int k = 0; k < half; k++) {
        int tw_idx = k * step;   /* index into twiddle[0..full_n/2-1] */

        float wr = twiddle_real[tw_idx];
        float wi = twiddle_imag[tw_idx];

        float ur = x_real[k];
        float ui = x_imag[k];
        float vr = x_real[k + half];
        float vi = x_imag[k + half];

        float tr = wr * vr - wi * vi;
        float ti = wr * vi + wi * vr;

        x_real[k]        = ur + tr;
        x_imag[k]        = ui + ti;
        x_real[k + half] = ur - tr;
        x_imag[k + half] = ui - ti;
    }
}

void butterfly_recursive(float *x_real, float *x_imag,
                          float *twiddle_real, float *twiddle_imag, int n) {
    /* full_n == n at the top level; the helper tracks the stride */
    butterfly_recursive_helper(x_real, x_imag, twiddle_real, twiddle_imag, n, n);
}

void fft_1d_recursive(float *x_real, float *x_imag,
                      float *twiddle_real, float *twiddle_imag, int n) {
    bit_reverse_array(x_real, x_imag, n);
    butterfly_recursive(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

static void transpose(float **matrix, int rows, int cols) {
    if (matrix == NULL) return;
    if (rows != cols) return;
    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < cols; j++) {
            float tmp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tmp;
        }
    }
}

void fft_2d(float **x_real, float **x_imag,
            float *twiddle_real, float *twiddle_imag,
            int rows, int cols, bool test, bool inverse) {
                if (!inverse)
    generate_twiddle_factors(twiddle_real, twiddle_imag, cols);
    /* --- Row FFTs (each row has 'cols' elements) --- */
    for (int r = 0; r < rows; r++)
        fft_1d_iterative(x_real[r], x_imag[r], twiddle_real, twiddle_imag, cols);

    if (test) {
        printf("Intermediate (after row FFTs):\n");
        for (int i = 0; i < rows && i < 4; i++) {
            for (int j = 0; j < cols && j < 4; j++)
                printf("(%.2f+%.2fi) ", x_real[i][j], x_imag[i][j]);
            printf("\n");
        }
    }

    //Transpose 
    transpose(x_real, rows, cols);
    transpose(x_imag, rows, cols);

    /* Column FFTs (each row after transpose has 'rows' elements) */
    for (int r = 0; r < cols; r++)
        fft_1d_iterative(x_real[r], x_imag[r], twiddle_real, twiddle_imag, rows);

    /*  Transpose back  */
    transpose(x_real, cols, rows);
    transpose(x_imag, cols, rows);
}

/* 
 2D INVERSE FFT
 */
void ifft_2d(float **x_real, float **x_imag,
             float *twiddle_real, float *twiddle_imag,
             int rows, int cols, bool test) {

    int max_dim = (rows > cols) ? rows : cols;
    generate_inverse_twiddle_factors(twiddle_real, twiddle_imag, max_dim);
    fft_2d(x_real, x_imag, twiddle_real, twiddle_imag, rows, cols, test, true);

    /* Normalize by N = rows * cols */
    float norm = 1.0f / (float)(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            x_real[i][j] *= norm;
            x_imag[i][j] *= norm;
        }
}


//  * GAUSSIAN HIGH-PASS FILTER
//  * H(i,j) = 1 - exp( -D²(i,j) / (2*D0²) )
//  * D(i,j) = sqrt( i_shifted² + j_shifted² )
//  * FFT output wraps around so we account for this.

void create_highpass_filter(float **filter, int rows, int cols, float cutoff) {
    float two_d0_sq = 2.0f * cutoff * cutoff;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            /* Shift frequencies so DC is at center */
            float fi = (float)(i < rows/2 ? i : i - rows);
            float fj = (float)(j < cols/2 ? j : j - cols);
            float d_sq = fi * fi + fj * fj;
            filter[i][j] = 1.0f - my_exp(-d_sq / two_d0_sq);
        }
    }
}

/* 
 * EDGE DETECTION
  1. Create high-pass filter
  2. Multiply FFT result by filter (element-wise complex * real)
  3. Inverse FFT back to spatial domain
 */
void edge_detection(float **x_real, float **x_imag,
                    float *twiddle_real, float *twiddle_imag,
                    float **filter, int rows, int cols, float cutoff) {
    /* Build filter */
    create_highpass_filter(filter, rows, cols, cutoff);

    /* Multiply FFT result by filter */
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            x_real[i][j] *= filter[i][j];
            x_imag[i][j] *= filter[i][j];
        }

    // Inverse FFT 
    ifft_2d(x_real, x_imag, twiddle_real, twiddle_imag, rows, cols, false);
}


// HELPER: next power of 
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

 // HELPER: comparison function for qsort

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}


 // HELPER: allocate 2D float array

static float** alloc_2d(int rows, int cols) {
    float **m = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++)
        m[i] = (float*)calloc(cols, sizeof(float));
    return m;
}

static void free_2d(float **m, int rows) {
    for (int i = 0; i < rows; i++) free(m[i]);
    free(m);
}


static void process_image(const char *input_path,
                           const char *fft_out_path,
                           const char *edge_out_path,
                           float cutoff) {
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 0);
    if (!img) {
        printf("Failed to load image: %s\n", input_path);
        return;
    }
    printf("Loaded %s: %dx%d, %d channels\n", input_path, width, height, channels);

    /* Pad to next power of 2 for FFT (square) */
    int pw = next_pow2(width);
    int ph = next_pow2(height);
    int dim = (pw > ph) ? pw : ph;   /* square: pad both to same size */
    pw = dim; ph = dim;

    /* Allocate padded matrices */
    float **x_real = alloc_2d(ph, pw);
    float **x_imag = alloc_2d(ph, pw);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float gray;
            if (channels == 1) {
                gray = img[r * width + c] / 255.0f;
            } else {
                unsigned char *px = &img[(r * width + c) * channels];
                gray = (0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2]) / 255.0f;
            }
            x_real[r][c] = gray;
        }
    }

    float *twiddle_real = (float*)malloc((dim / 2) * sizeof(float));
    float *twiddle_imag = (float*)malloc((dim / 2) * sizeof(float));
    generate_twiddle_factors(twiddle_real, twiddle_imag, dim);

    /*  Forward 2D FFT */
    fft_2d(x_real, x_imag, twiddle_real, twiddle_imag, ph, pw, false, false);

    {
        float max_mag = 0.0f;
        float *mag = (float*)malloc(ph * pw * sizeof(float));

        for (int r = 0; r < ph; r++)
            for (int c = 0; c < pw; c++) {
                float m = sqrtf(x_real[r][c]*x_real[r][c] + x_imag[r][c]*x_imag[r][c]);
                mag[r * pw + c] = logf(1.0f + m);
                if (mag[r * pw + c] > max_mag) max_mag = mag[r * pw + c];
            }

        unsigned char *fft_out = (unsigned char*)malloc(height * width);
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                fft_out[r * width + c] = (unsigned char)(255.0f * mag[r * pw + c] / (max_mag + 1e-6f));

        stbi_write_png(fft_out_path, width, height, 1, fft_out, width);
        printf("Saved FFT magnitude: %s\n", fft_out_path);
        free(fft_out);
        free(mag);
    }

    /* ---- Edge Detection ----
     * x_real/x_imag currently hold the 2D FFT result.
     * edge_detection applies the high-pass filter then calls ifft_2d.
      */
    float **filter = alloc_2d(ph, pw);
    edge_detection(x_real, x_imag, twiddle_real, twiddle_imag, filter, ph, pw, cutoff);
    /* x_real/x_imag now contains the spatial-domain edge image */

    {
        int npix = height * width;
        float *vals = (float*)malloc(npix * sizeof(float));
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                vals[r * width + c] = fabsf(x_real[r][c]);

        float *sorted = (float*)malloc(npix * sizeof(float));
        memcpy(sorted, vals, npix * sizeof(float));
        qsort(sorted, npix, sizeof(float), float_cmp);
        float clip_max = sorted[(int)(0.95f * (npix - 1))];
        if (clip_max < 1e-6f) clip_max = 1e-6f;
        free(sorted);

        const float GAMMA      = 1.80f;   
        const float THRESHOLD  = 0.28f;
        const float BRIGHTNESS = 0.38f;   

        unsigned char *edge_out = (unsigned char*)malloc(npix);
        for (int i = 0; i < npix; i++) {
            float v = vals[i] / clip_max;
            if (v > 1.0f) v = 1.0f;
            if (v < THRESHOLD) v = 0.0f;    
            else v = powf(v, GAMMA) * BRIGHTNESS;
            edge_out[i] = (unsigned char)(255.0f * v);
        }

        stbi_write_png(edge_out_path, width, height, 1, edge_out, width);
        printf("Saved edge detection: %s\n", edge_out_path);
        free(edge_out);
        free(vals);
    }

    /* Cleanup */
    free_2d(x_real, ph);
    free_2d(x_imag, ph);
    free_2d(filter, ph);
    free(twiddle_real);
    free(twiddle_imag);
    stbi_image_free(img);
}


int main() {
    process_image("input images/8x8.png", "output images/fft_8x8.png", "output images/edges_8x8.png", 120.0f);
    process_image("input images/8x8-2.png", "output images/fft_8x8-2.png", "output images/edges_8x8-2.png", 120.0f);
    process_image("input images/16x16.png", "output images/fft_16x16.png", "output images/edges_16x16.png", 120.0f);
    process_image("input images/OIP.png", "output images/fft_OIP.png", "output images/edges_OIP.png", 120.0f);
    process_image("input images/R.png", "output images/fft_R.png", "output images/edges_R.png", 120.0f);
    return 0;
}