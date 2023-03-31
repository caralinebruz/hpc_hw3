/* Jacobi smoothing to solve -u''=f
 * Global vector has N inner unknowns.
 * gcc -fopenmp -lm 04-omp-jacobi.c && ./a.out 200000 100
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// #ifdef _OPENMP
// #include <omp.h>
// #else
// #include "utils.h"
// #endif

// #ifdef __APPLE__ 
// #include "utils.h"
// #else
// #include <omp.h>
// #endif

#include "utils.h"

// *** UTILS
void printy(int N, double* q) {
	// printing f
	for (int i = 0; i < N; i++) {
	    std::cout << q[i] << ' ';
	}
	std::cout << std::endl;	
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *u, int N, double invhsq)
{
	int i;
	double tmp, res = 0.0;
#pragma omp parallel for reduction (+:res)
	for (i = 1; i <= N; i++){
		tmp = ((2.0*u[i] - u[i-1] - u[i+1]) * invhsq - 1);
		res += tmp * tmp;
	}
	return sqrt(res);
}


int main(int argc, char * argv[])
{
	int i, N, iter, max_iters;

	// sscanf(argv[1], "%d", &N);
	// sscanf(argv[2], "%d", &max_iters);
	N = 5;
	max_iters = 1000;

	Timer tt;
	tt.tic();

	/* Allocation of vectors, including left and right ghost points */
	double* u    = (double *) calloc(sizeof(double), N+2);
	double* unew = (double *) calloc(sizeof(double), N+2);

	double h = 1.0 / (N + 1);
	double hsq = h*h;
	double invhsq = 1./hsq;
	double initial_residual, residual, tol = 1e-5;

	printy(N, u);

	/* initial residual */
	initial_residual = compute_residual(u, N, invhsq);
	printf("Initial Residual: %g\n",initial_residual);

	residual = initial_residual;
	u[0] = u[N+1] = 0.0;
	double omega = 1.0; //2./3;


	// do 1D jacobi until convergence
	for (iter = 0; iter < max_iters && residual/initial_residual > tol; iter++) {

		/* Jacobi step for all the inner points */
		for (i = 1; i <= N; i++){
			unew[i] =  u[i] + omega * 0.5 * (hsq + u[i - 1] + u[i + 1] - 2*u[i]);
		}

		/* flip pointers; that's faster than memcpy  */
		// memcpy(u,unew,(N+2)*sizeof(double));
		double* utemp = u;
		u = unew;
		unew = utemp;
		if (0 == (iter % 1)) {
			residual = compute_residual(u, N, invhsq);
			printf("Iter %d: Residual: %g\n", iter, residual);
		}
	}

	/* Clean up */
	free(u);
	free(unew);

	/* timing */
	double t = tt.toc();
	printf("Time elapsed is %f.\n", t);
	return 0;
}