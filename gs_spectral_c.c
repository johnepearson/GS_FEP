/*
 * Gray-Scott pseudo-spectral (ETD1) solver, C implementation.
 *
 * Reduces the per-step FFT count from 6 (as done naively/in the
 * existing Python PseudoSpectralGS, which re-transforms u and v
 * forward every step even though it already has u_hat/v_hat from the
 * previous step's update) to 3, by:
 *
 *   1. keeping the state as (u_hat, v_hat) in Fourier space between
 *      steps, rather than round-tripping through real space and back
 *      -- this alone removes 2 of the 6 (the forward transforms of u
 *      and v, which are redundant: u_hat/v_hat from the previous
 *      step's update already ARE their forward transform).
 *   2. noting that the reaction term only needs ONE forward
 *      transform, not two: writing w = u*v^2, the two reaction terms
 *      are Ru = -w + F(1-u) and Rv = w - (F+k)v, so in Fourier space
 *      Ru_hat = -w_hat + F(N - u_hat) [N = n*n at k=0 only, since
 *      that's the unnormalized transform of the constant field 1] and
 *      Rv_hat = w_hat - (F+k) v_hat -- both are cheap linear
 *      combinations of w_hat (one forward FFT) and the u_hat, v_hat
 *      already in hand from step 1, with no separate FFT of Ru or Rv
 *      needed.
 *
 * That leaves exactly 3 FFTs per step: inverse-transform u_hat and
 * v_hat to real space (2), forward-transform w = u*v^2 (1).
 *
 * Uses FFTW's real-to-complex / complex-to-real 2D transforms
 * (r2c/c2r), which is what scipy/numpy's rfft2/irfft2 wrap too, so
 * the spectral layout (shape (n, n/2+1), same frequency ordering)
 * matches the existing Python implementation exactly -- checkpoints
 * are interchangeable between the two.
 *
 * FFTW's transforms are unnormalized in both directions (forward AND
 * inverse), unlike numpy/scipy's convention (forward unnormalized,
 * inverse divided by n*n). We match the numpy/scipy convention
 * explicitly by dividing by n*n after every inverse transform here,
 * so amplitudes (and therefore Eu/Ev/phi1_u/phi1_v, computed the same
 * way as the Python code) are identical.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <unistd.h>

static int _threads_inited = 0;
static void ensure_threads_inited(void) {
    if (!_threads_inited) {
        fftw_init_threads();
        long n;
        const char *env = getenv("GS_FFT_THREADS");
        if (env && atol(env) > 0) {
            n = atol(env);
        } else {
            n = sysconf(_SC_NPROCESSORS_ONLN);
            if (n < 1) n = 1;
        }
        fftw_plan_with_nthreads((int)n);
        fprintf(stderr, "[gs_spectral_c] FFTW threads = %ld (set GS_FFT_THREADS to override)\n", n);
        _threads_inited = 1;
    }
}

typedef struct {
    int n;
    double L, dx, Du, Dv, F, k, dt;
    double time;
    long step_count;

    int nk;  /* n/2 + 1, the reduced (rfft) axis length */

    fftw_complex *uhat, *vhat;      /* persistent state, size n*nk */
    fftw_complex *uhat_tmp, *vhat_tmp; /* scratch copies for c2r (which
                                          destroys its input) */
    fftw_complex *what;             /* scratch for forward transform of w */
    double *ureal, *vreal, *wreal;  /* real-space scratch, size n*n */

    double *Eu, *Ev, *phi1_u, *phi1_v; /* precomputed factors, size n*nk */

    int dealias;                    /* 0/1: apply the 2/3-rule mask below */
    unsigned char *dealias_mask;    /* size n*nk; 1 = keep, 0 = zero out.
                                        Standard separable (per-axis) 2/3
                                        rule: a mode survives only if BOTH
                                        its kx and ky indices are within
                                        the lowest 2/3 of that axis's
                                        represented wavenumbers. Applied to
                                        what (the transform of the cubic
                                        nonlinearity w=u*v^2) once per
                                        step, right after its forward FFT
                                        and before it is combined into
                                        Ru_hat/Rv_hat -- i.e. it truncates
                                        the aliased high-wavenumber content
                                        the nonlinear product generates,
                                        before that content re-enters the
                                        next step's evolution. This is the
                                        one deliberate difference from the
                                        existing (un-dealiased) solver;
                                        everything else is identical. */

    fftw_plan plan_inv_u, plan_inv_v; /* c2r, out-of-place */
    fftw_plan plan_fwd_w;             /* r2c, out-of-place */
} gs_sim;

static void precompute_dealias_mask(gs_sim *s) {
    int n = s->n, nk = s->nk;
    s->dealias_mask = (unsigned char*)malloc(sizeof(unsigned char) * n * nk);
    int cutoff = n / 3;  /* standard 2/3-rule cutoff, integer division */
    for (int i = 0; i < n; i++) {
        int kx_idx = (i <= n / 2) ? i : i - n;
        int keep_kx = (abs(kx_idx) <= cutoff);
        for (int j = 0; j < nk; j++) {
            int keep_ky = (j <= cutoff);
            s->dealias_mask[i * nk + j] = (unsigned char)(keep_kx && keep_ky);
        }
    }
}

static void precompute_factors(gs_sim *s) {
    int n = s->n, nk = s->nk;
    s->Eu = (double*)malloc(sizeof(double) * n * nk);
    s->Ev = (double*)malloc(sizeof(double) * n * nk);
    s->phi1_u = (double*)malloc(sizeof(double) * n * nk);
    s->phi1_v = (double*)malloc(sizeof(double) * n * nk);

    for (int i = 0; i < n; i++) {
        int kx_idx = (i <= n / 2) ? i : i - n;
        double kx = 2.0 * M_PI * kx_idx / (n * s->dx);
        for (int j = 0; j < nk; j++) {
            double ky = 2.0 * M_PI * j / (n * s->dx);
            double K2 = kx * kx + ky * ky;
            double Lu = -s->Du * K2;
            double Lv = -s->Dv * K2;
            double Eu = exp(Lu * s->dt);
            double Ev = exp(Lv * s->dt);
            double phi1_u, phi1_v;
            if (K2 == 0.0) {
                phi1_u = 1.0;
                phi1_v = 1.0;
            } else {
                phi1_u = (Eu - 1.0) / (Lu * s->dt);
                phi1_v = (Ev - 1.0) / (Lv * s->dt);
            }
            int idx = i * nk + j;
            s->Eu[idx] = Eu;
            s->Ev[idx] = Ev;
            s->phi1_u[idx] = phi1_u;
            s->phi1_v[idx] = phi1_v;
        }
    }
}

void *gs_create(int n, double L, double Du, double Dv, double F, double k, double dt, int dealias) {
    ensure_threads_inited();
    gs_sim *s = (gs_sim*)calloc(1, sizeof(gs_sim));
    s->n = n;
    s->L = L;
    s->dx = L / n;
    s->Du = Du;
    s->Dv = Dv;
    s->F = F;
    s->k = k;
    s->dt = dt;
    s->time = 0.0;
    s->step_count = 0;
    s->nk = n / 2 + 1;
    s->dealias = dealias;

    int n2 = n * n;
    int nnk = n * s->nk;

    s->uhat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nnk);
    s->vhat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nnk);
    s->uhat_tmp = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nnk);
    s->vhat_tmp = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nnk);
    s->what = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nnk);
    s->ureal = (double*)fftw_malloc(sizeof(double) * n2);
    s->vreal = (double*)fftw_malloc(sizeof(double) * n2);
    s->wreal = (double*)fftw_malloc(sizeof(double) * n2);
    memset(s->uhat, 0, sizeof(fftw_complex) * nnk);
    memset(s->vhat, 0, sizeof(fftw_complex) * nnk);

    /* out-of-place plans; FFTW_MEASURE would overwrite the scratch
       buffers during planning, which is fine since they're scratch,
       but costs a bit of startup time -- ESTIMATE is fine here since
       plans are created once per simulation, not per step. */
    s->plan_inv_u = fftw_plan_dft_c2r_2d(n, n, s->uhat_tmp, s->ureal, FFTW_ESTIMATE);
    s->plan_inv_v = fftw_plan_dft_c2r_2d(n, n, s->vhat_tmp, s->vreal, FFTW_ESTIMATE);
    s->plan_fwd_w = fftw_plan_dft_r2c_2d(n, n, s->wreal, s->what, FFTW_ESTIMATE);

    precompute_factors(s);
    precompute_dealias_mask(s);
    return (void*)s;
}

void gs_destroy(void *sim) {
    gs_sim *s = (gs_sim*)sim;
    fftw_destroy_plan(s->plan_inv_u);
    fftw_destroy_plan(s->plan_inv_v);
    fftw_destroy_plan(s->plan_fwd_w);
    fftw_free(s->uhat); fftw_free(s->vhat);
    fftw_free(s->uhat_tmp); fftw_free(s->vhat_tmp);
    fftw_free(s->what);
    fftw_free(s->ureal); fftw_free(s->vreal); fftw_free(s->wreal);
    free(s->Eu); free(s->Ev); free(s->phi1_u); free(s->phi1_v);
    free(s->dealias_mask);
    free(s);
}

/* Set state from real-space u, v fields (row-major, n x n). Requires
   one forward FFT each of u and v -- only paid once, at setup. */
void gs_set_field(void *sim, const double *u, const double *v) {
    gs_sim *s = (gs_sim*)sim;
    int n = s->n, n2 = n * n, nnk = n * s->nk;
    memcpy(s->ureal, u, sizeof(double) * n2);
    fftw_plan p = fftw_plan_dft_r2c_2d(n, n, s->ureal, s->uhat, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    memcpy(s->vreal, v, sizeof(double) * n2);
    p = fftw_plan_dft_r2c_2d(n, n, s->vreal, s->vhat, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    (void)nnk;
}

/* Retrieve current real-space fields (does the 2 inverse FFTs needed
   to materialize them; does not advance the simulation or touch the
   persistent Fourier-space state). */
void gs_get_field(void *sim, double *u_out, double *v_out) {
    gs_sim *s = (gs_sim*)sim;
    int n = s->n, n2 = n * n, nnk = n * s->nk;
    memcpy(s->uhat_tmp, s->uhat, sizeof(fftw_complex) * nnk);
    fftw_execute(s->plan_inv_u);
    memcpy(s->vhat_tmp, s->vhat, sizeof(fftw_complex) * nnk);
    fftw_execute(s->plan_inv_v);
    double norm = 1.0 / (double)n2;
    for (int idx = 0; idx < n2; idx++) {
        u_out[idx] = s->ureal[idx] * norm;
        v_out[idx] = s->vreal[idx] * norm;
    }
}

static void gs_step1(gs_sim *s) {
    int n = s->n, nk = s->nk, n2 = n * n, nnk = n * nk;
    double norm = 1.0 / (double)n2;

    /* 2 inverse FFTs: u_hat, v_hat -> real space */
    memcpy(s->uhat_tmp, s->uhat, sizeof(fftw_complex) * nnk);
    fftw_execute(s->plan_inv_u);
    memcpy(s->vhat_tmp, s->vhat, sizeof(fftw_complex) * nnk);
    fftw_execute(s->plan_inv_v);

    /* w = u * v^2, real space, single fused pass; also apply the
       normalization here so u/v aren't touched twice */
    for (int idx = 0; idx < n2; idx++) {
        double u = s->ureal[idx] * norm;
        double v = s->vreal[idx] * norm;
        s->ureal[idx] = u;  /* store normalized value back for reuse below */
        s->vreal[idx] = v;
        s->wreal[idx] = u * v * v;
    }

    /* 1 forward FFT: w -> what */
    fftw_execute(s->plan_fwd_w);

    /* 2/3-rule dealiasing: zero the high-wavenumber third of what
       (the cubic nonlinearity's transform) before it feeds back into
       the update, if enabled. No-op, and bit-identical to the
       un-dealiased solver, when s->dealias == 0. */
    if (s->dealias) {
        for (int idx = 0; idx < n * nk; idx++) {
            if (!s->dealias_mask[idx]) {
                s->what[idx][0] = 0.0;
                s->what[idx][1] = 0.0;
            }
        }
    }

    /* Ru_hat = -what + F*(N*delta_{k=0} - uhat)
       Rv_hat = what - (F+k)*vhat
       uhat_new = Eu*uhat + dt*phi1_u*Ru_hat
       vhat_new = Ev*vhat + dt*phi1_v*Rv_hat */
    double F = s->F, kk = s->k, dt = s->dt;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nk; j++) {
            int idx = i * nk + j;
            double what_re = s->what[idx][0];
            double what_im = s->what[idx][1];
            double uhat_re = s->uhat[idx][0];
            double uhat_im = s->uhat[idx][1];
            double vhat_re = s->vhat[idx][0];
            double vhat_im = s->vhat[idx][1];

            double Ru_re = -what_re - F * uhat_re;
            double Ru_im = -what_im - F * uhat_im;
            double Rv_re = what_re - (F + kk) * vhat_re;
            double Rv_im = what_im - (F + kk) * vhat_im;
            if (i == 0 && j == 0) {
                Ru_re += F * (double)n2;
            }

            double Eu = s->Eu[idx], Ev = s->Ev[idx];
            double p1u = s->phi1_u[idx], p1v = s->phi1_v[idx];

            s->uhat[idx][0] = Eu * uhat_re + dt * p1u * Ru_re;
            s->uhat[idx][1] = Eu * uhat_im + dt * p1u * Ru_im;
            s->vhat[idx][0] = Ev * vhat_re + dt * p1v * Rv_re;
            s->vhat[idx][1] = Ev * vhat_im + dt * p1v * Rv_im;
        }
    }

    s->time += dt;
    s->step_count += 1;
}

void gs_run(void *sim, int n_steps) {
    gs_sim *s = (gs_sim*)sim;
    for (int i = 0; i < n_steps; i++) gs_step1(s);
}

double gs_get_time(void *sim) { return ((gs_sim*)sim)->time; }
long gs_get_step_count(void *sim) { return ((gs_sim*)sim)->step_count; }
void gs_set_time(void *sim, double t, long step_count) {
    gs_sim *s = (gs_sim*)sim;
    s->time = t;
    s->step_count = step_count;
}
