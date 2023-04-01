/* Jacobi smoothing to solve -u''=f
 * Global vector has N inner unknowns.
 * gcc -fopenmp -lm 04-omp-jacobi.c && ./a.out 200000 100
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils.h"
#endif

double** u;
double** unew;

void free_mems(int N) {
	// Free the memory after the use of array

 	for (int i = 0; i < N+2; i++) {
		delete[] u[i];
 	}
 	delete[] u;

 	for (int i = 0; i < N+2; i++) {
		delete[] unew[i];
 	}
 	delete[] unew;
}

// *** UTILS
void printy(int N, double** q) {

	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
	    	printf("%2lf ", q[i][j]);
		}
		printf("\n");
	}
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **u, int N, double invhsq, double hsq)
{
	int i,j;
	double tmp, res = 0.0;
	#pragma omp parallel for reduction (+:res)
	for (i = 1; i <= N; i++){
		for (j=1; j <= N; j++) {

			tmp = ( (-u[i-1][j] - u[i][j-1] + 4.0*u[i][j] -u[i+1][j] -u[i][j+1] ) * invhsq - 1); // working but valyes are off
			res += tmp * tmp;
		}
	}
	return sqrt(res);
}


int main(int argc, char * argv[])
{
	int i, j, N, iter, max_iters;
	sscanf(argv[1], "%d", &N);
	sscanf(argv[2], "%d", &max_iters);

	// test omp
	#pragma omp parallel    //pragma is ignored if the compiler does not know it
	{
	#if defined(_OPENMP)
		printf("hello world from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
	#else
		printf("hello world from the only thread around\n");
	#endif
	}

	#ifdef _OPENMP
		double t = omp_get_wtime();
	#else
		Timer tt;
		tt.tic();
	#endif

	/* Allocation of vectors, including left and right ghost points */
	u = new double*[N+2];
	unew = new double*[N+2];

	// By using a loop, we will allocate memory to each row of the 2D array.
	for (int i = 0; i < N+2; i++) {
		u[i] = new double[N+2];
		unew[i] = new double[N+2];

		// while we are here, init ends to 0.0
		u[i][0] = 0.0;
		u[i][N+1] = 0.0;
	}
	// then set top and bottom rows to be 0.0
	for (int j=0;j<N+2;j++) {
		u[0][j] = 0.0;
		u[N+1][j] = 0.0;
	}

	double h = 1.0 / (N + 1);
	double hsq = h*h;
	double invhsq = 1./hsq;
	double initial_residual, residual, tol = 1e-5;

	/* initial residual */
	initial_residual = compute_residual(u, N, invhsq, hsq);
	printf("Initial Residual: %g\n",initial_residual);
	residual = initial_residual;

	// // do 1D jacobi until convergence
	for (iter = 0; iter < max_iters && residual/initial_residual > tol; iter++) {

		/* Jacobi step for all the inner points */
		for (i = 1; i <= N; i++){
			for (j=1; j<=N; j++) {

				unew[i][j] = u[i][j] + 0.25 * (hsq + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1] -4.0*u[i][j]); // working but off values by 0.1
			}
			// unew[i] =  u[i] + omega * 0.5 * (hsq + u[i - 1] + u[i + 1] - 2*u[i]); // 1d case
		}

		// printy(N,unew);

		/* flip pointers; that's faster than memcpy  */
		// memcpy(u,unew,(N+2)*sizeof(double));
		double** utemp = u;
		u = unew;
		unew = utemp;
		if (0 == (iter % 1)) {
			residual = compute_residual(u, N, invhsq, hsq);
			printf("Iter %d: Residual: %g\n", iter, residual);
		}
	}

	printy(N,unew);

	/* Clean up */
	free_mems(N);

	/* timing */
	#ifdef _OPENMP
		t = omp_get_wtime() - t;
	#else
		double t = tt.toc();
	#endif
	printf("Time elapsed is %f.\n", t);
	return 0;
}