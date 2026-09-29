#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <immintrin.h>

#define WIDTH 80
#define HEIGHT 24
#define PI 3.14159265358979323846f

static const char SHADES[] = ".,-~:;=!*#$@";

// Fast polynomial approximation for 8-wide sin/cos (Taylor/Remez)
static inline void sincos256(__m256 x, __m256 *s, __m256 *c) {
    __m256 inv_2pi = _mm256_set1_ps(0.159154943f);
    __m256 pi2     = _mm256_set1_ps(6.283185307f);

    // Normalize to [-pi, pi]
    __m256 y = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(x, inv_2pi), 
               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), pi2));

    __m256 y2 = _mm256_mul_ps(y, y);

    // Polynomial coefficients
    __m256 s3 = _mm256_set1_ps(-1.66666666e-1f);
    __m256 s5 = _mm256_set1_ps( 8.33333333e-3f);
    __m256 s7 = _mm256_set1_ps(-1.98412698e-4f);

    __m256 c2 = _mm256_set1_ps(-0.5f);
    __m256 c4 = _mm256_set1_ps( 4.16666667e-2f);
    __m256 c6 = _mm256_set1_ps(-1.38888889e-3f);

    // Taylor series: sin(y) ≈ y * (1 + y^2*(s3 + y^2*(s5 + y^2*s7)))
    __m256 s_poly = _mm256_fmadd_ps(y2, s7, s5);
    s_poly = _mm256_fmadd_ps(y2, s_poly, s3);
    *s = _mm256_fmadd_ps(_mm256_mul_ps(y, y2), s_poly, y);

    // Taylor series: cos(y) ≈ 1 + y^2*(c2 + y^2*(c4 + y^2*c6))
    __m256 c_poly = _mm256_fmadd_ps(y2, c6, c4);
    c_poly = _mm256_fmadd_ps(y2, c_poly, c2);
    *c = _mm256_fmadd_ps(y2, c_poly, _mm256_set1_ps(1.0f));
}

int main(void) {
    float A = 0.0f, B = 0.0f;
    char b[WIDTH * HEIGHT];
    float z[WIDTH * HEIGHT];

    // Hide cursor & clear screen
    printf("\x1b[?25l\x1b[2J");

    const float R1 = 1.0f;
    const float R2 = 2.0f;
    const float K2 = 5.0f;
    const float K1 = WIDTH * K2 * 3.0f / (8.0f * (R1 + R2));
    const float dtheta = 0.07f;
    const float dphi = 0.02f;

    // Vector lane offsets [0..7] * dtheta
    const __m256 lane_idx = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256 v_dtheta_step = _mm256_mul_ps(lane_idx, _mm256_set1_ps(dtheta));

    while (1) {
        memset(b, ' ', sizeof(b));
        memset(z, 0, sizeof(z));

        float sinA = sinf(A), cosA = cosf(A);
        float sinB = sinf(B), cosB = cosf(B);

        __m256 v_sinA = _mm256_set1_ps(sinA), v_cosA = _mm256_set1_ps(cosA);
        __m256 v_sinB = _mm256_set1_ps(sinB), v_cosB = _mm256_set1_ps(cosB);
        __m256 v_R1   = _mm256_set1_ps(R1);
        __m256 v_R2   = _mm256_set1_ps(R2);
        __m256 v_K1   = _mm256_set1_ps(K1);
        __m256 v_K2   = _mm256_set1_ps(K2);

        for (float phi = 0.0f; phi < 2.0f * PI; phi += dphi) {
            float sinphi = sinf(phi), cosphi = cosf(phi);
            __m256 v_sinphi = _mm256_set1_ps(sinphi);
            __m256 v_cosphi = _mm256_set1_ps(cosphi);

            // Process theta in chunks of 8
            for (float theta = 0.0f; theta < 2.0f * PI; theta += 8.0f * dtheta) {
                __m256 v_theta = _mm256_add_ps(_mm256_set1_ps(theta), v_dtheta_step);
                __m256 v_sintheta, v_costheta;
                sincos256(v_theta, &v_sintheta, &v_costheta);

                // Torus geometry
                __m256 circlex = _mm256_fmadd_ps(v_R1, v_costheta, v_R2);
                __m256 circley = _mm256_mul_ps(v_R1, v_sintheta);

                // 3D rotations
                __m256 x = _mm256_mul_ps(circlex, _mm256_fmsub_ps(v_cosB, v_cosphi, _mm256_mul_ps(v_sinA, _mm256_mul_ps(v_sinB, v_sinphi))));
                x = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_mul_ps(circley, v_cosA), v_sinB));

                __m256 y = _mm256_mul_ps(circlex, _mm256_fmadd_ps(v_sinB, v_cosphi, _mm256_mul_ps(_mm256_mul_ps(v_sinA, v_cosB), v_sinphi)));
                y = _mm256_add_ps(y, _mm256_mul_ps(_mm256_mul_ps(circley, v_cosA), v_cosB));

                __m256 z_depth = _mm256_fmadd_ps(_mm256_mul_ps(circlex, v_cosA), v_sinphi, _mm256_fmadd_ps(circley, v_sinA, v_K2));
                __m256 ooz = _mm256_rcp_ps(z_depth); // 1 / z reciprocal estimate

                // Projected 2D screen coordinates
                __m256 xp = _mm256_fmadd_ps(x, _mm256_mul_ps(v_K1, ooz), _mm256_set1_ps((float)(WIDTH / 2)));
                __m256 yp = _mm256_fmadd_ps(_mm256_mul_ps(y, _mm256_mul_ps(v_K1, ooz)), _mm256_set1_ps(0.5f), _mm256_set1_ps((float)(HEIGHT / 2)));

                // Surface normal / illumination L
                __m256 L = _mm256_mul_ps(_mm256_fmsub_ps(v_sintheta, v_sinA, _mm256_mul_ps(v_costheta, _mm256_mul_ps(v_cosA, v_sinphi))), v_cosB);
                L = _mm256_sub_ps(L, _mm256_mul_ps(v_costheta, _mm256_mul_ps(v_cosA, v_cosphi)));
                L = _mm256_sub_ps(L, _mm256_mul_ps(v_sintheta, v_cosA));
                L = _mm256_add_ps(L, _mm256_mul_ps(_mm256_mul_ps(v_sinA, v_sinB), _mm256_mul_ps(v_costheta, v_cosphi)));

                // Extract SIMD registers to scalar for raster buffer scatter
                float xp_arr[8], yp_arr[8], ooz_arr[8], L_arr[8];
                _mm256_storeu_ps(xp_arr, xp);
                _mm256_storeu_ps(yp_arr, yp);
                _mm256_storeu_ps(ooz_arr, ooz);
                _mm256_storeu_ps(L_arr, L);

                for (int i = 0; i < 8; i++) {
                    // Check the floating-point coordinates before converting:
                    // a value such as -0.4 truncates to 0 and would otherwise
                    // incorrectly draw a point just outside the left/top edge.
                    if (!(xp_arr[i] >= 0.0f && xp_arr[i] < (float)WIDTH &&
                          yp_arr[i] >= 0.0f && yp_arr[i] < (float)HEIGHT &&
                          ooz_arr[i] > 0.0f)) {
                        continue;
                    }

                    int px = (int)xp_arr[i];
                    int py = (int)yp_arr[i];

                    int idx = py * WIDTH + px;
                    if (ooz_arr[i] > z[idx]) {
                        z[idx] = ooz_arr[i];
                        int lum = (int)(L_arr[i] * 8.0f);
                        b[idx] = SHADES[lum > 0 ? (lum < 11 ? lum : 11) : 0];
                    }
                }
            }
        }

        // Draw entire buffer in a single write call
        printf("\x1b[H");
        for (int k = 0; k < HEIGHT; k++) {
            fwrite(&b[k * WIDTH], 1, WIDTH, stdout);
            putchar('\n');
        }
        fflush(stdout);

        // Rotation angles
        A += 0.07f;
        B += 0.035f;

        usleep(32000); // ~30 FPS
    }

    return 0;
}
