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
	    	// cout << A[i][j] << '\t';
	    	printf("%2lf ", q[i][j]);
		}
		std::cout << std::endl;
	}
	
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **u, int N, double invhsq)
{
	int i,j;
	double tmp, res = 0.0;
// #pragma omp parallel for reduction (+:res)

	// for (i = 1; i <= N; i++){

	// 	for (j = 1; j <= N; j++) {

	// 		// 1D
	// 		// tmp = ((2.0*u[i] - u[i-1] - u[i+1]) * invhsq - 1);

	// 		// i will need to add the y-term here when i figure 
	// 		// tmp = ( (2.0*u[i][j] - u[i-1][j] - u[i+1][j] )  * invhsq - 1);
	// 		tmp = (u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1] * invhsq - 1);

	// 		res += tmp * tmp;
	// 	}
	// }
	// return sqrt(res);


	// takes the largest difference between previous values in u and u_{k-1}
	// this is not necessarily the residual, but i could not figure out
	// how the residual should be calculated in the 2D case
	double residual = 0.0;
	double delta = 0.0;

	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {

			delta = fabs(unew[i][j] - u[i][j]);

			printf("delta is %f \n", delta);
			if (delta > residual) {
				residual = delta;
			}
		}
	}
	return residual;
}


int main(int argc, char * argv[])
{
	int i, j, x, y, N, iter, max_iters;

	// sscanf(argv[1], "%d", &N);
	// sscanf(argv[2], "%d", &max_iters);
	N = 5;
	max_iters = 1000;

	Timer tt;
	tt.tic();

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

	//double* u    = (double *) calloc(sizeof(double), ((N+2)*(N+2)));
	//double* unew = (double *) calloc(sizeof(double), ((N+2)*(N+2)));

	double h = 1.0 / (N + 1);
	double hsq = h*h;
	double invhsq = 1./hsq;
	double initial_residual, residual, tol = 1e-5;

	printy(N, u);



	// because of my residual calculation *** 
	/* Jacobi step for all the inner points */
	for (i = 1; i <= N; i++){
		for (j=1; j<=N; j++) {

			unew[i][j] = 0.25 * (hsq + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]);
		}
		// unew[i] =  u[i] + omega * 0.5 * (hsq + u[i - 1] + u[i + 1] - 2*u[i]);
	}

	printy(N,unew);
	/* flip pointers; that's faster than memcpy  */
	// memcpy(u,unew,(N+2)*sizeof(double));
	double** utemp = u;
	u = unew;
	unew = utemp;
	if (0 == (iter % 1)) {
		residual = compute_residual(u, N, invhsq);
		printf("Iter %d: Residual: %g\n", iter, residual);
	}
	/// ^^^^ becase of my residual calculation



	/* initial residual */
	initial_residual = compute_residual(u, N, invhsq);
	printf("Initial Residual: %g\n",initial_residual);

	residual = initial_residual;

	//double omega = 1.0; //2./3;
	double omega = 2.0;


	// do 1D gs until convergence
	for (iter = 0; iter < max_iters && residual/initial_residual > tol; iter++) {

		// gs step to update all red points (move in increments of 2)
		for (i = 1; i <= N; i++){

			if (i%2 == 0){
				// use even j
				for (j=2; j<=N;j+=2) {
					//printf("i %d: j: %d\n", i,j);
					unew[i][j] = 0.25 * (hsq + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]);
				}
			}
			else {
				// use odd j
				for (j=1; j<=N;j+=2) {
					//printf("i %d: j: %d\n", i,j);
					unew[i][j] = 0.25 * (hsq + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]);
				}
			}
		}
		printf("Red points updated:\n");
		printy(N,unew);

		// // gs step to update all black points
		for (i = 1; i <= N; i++){

			if (i%2 == 0){
				// use ODD j
				for (j=1; j<=N;j+=2) {
					//printf("i %d: j: %d\n", i,j);
					unew[i][j] = 0.25 * (hsq + unew[i-1][j] + unew[i][j-1] + unew[i+1][j] + unew[i][j+1]);
				}
			}
			else {
				// use EVEN j
				for (j=2; j<=N;j+=2) {
					//printf("i %d: j: %d\n", i,j);
					unew[i][j] = 0.25 * (hsq + unew[i-1][j] + unew[i][j-1] + unew[i+1][j] + unew[i][j+1]);
				}
			}
		}
		printf("Red and black points updated:\n");
		printy(N,unew);


		/* flip pointers; that's faster than memcpy  */
		// memcpy(u,unew,(N+2)*sizeof(double));
		double** utemp = u;
		u = unew;
		unew = utemp;
		if (0 == (iter % 1)) {
			residual = compute_residual(u, N, invhsq);
			printf("Iter %d: Residual: %g\n", iter, residual);
		}
	}

	/* Clean up */
	// free(u);
	// free(unew);
	free_mems(N);

	/* timing */
	double t = tt.toc();
	printf("Time elapsed is %f.\n", t);
	return 0;
}