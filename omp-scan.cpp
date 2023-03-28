#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
// #include <vector>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;

  for (long i = 1; i < n; i++) {

    // running sum, prev_value + next value in A

    // printf("\t + A[%i-1]: %lu \n", i, A[i-1]);

    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    // printf("\t = : %lu \n", prefix_sum[i]);
    
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();

  if (n == 0) return;
  prefix_sum[0] = 0;

  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel

  int NTHREADS = 6;
  int size_of_shared_storage = NTHREADS;
  int chunk_size = n/NTHREADS;

  // long offs = 0;
  // #pragma omp parallel for schedule(static,chunk_size) num_threads(numthreads) reduction(+:offs)
  //   for (long i = 1; i < n; i++) {

  //     printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);

  //     prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  //     offs =  prefix_sum[i];
  //     printf("Thread %d with acc %d.\n", omp_get_thread_num(), acc);
  // }
  // // printf("Thread %d with acc %d.\n", omp_get_thread_num(), acc);



  long* shared = (long*) malloc(size_of_shared_storage * sizeof(long));
  for (long i = 0; i < size_of_shared_storage; i++) shared[i] = 0;

  long acc = 0;
  #pragma omp parallel num_threads(NTHREADS)
  {

    int offs = 0;
    long* chunked_array = (long*) malloc(chunk_size * sizeof(long));
    chunked_array[0] = 0;

    
    #pragma omp for schedule(static,chunk_size) reduction(+:acc)
      for (long i = 1; i < chunk_size; i++) {

        printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);
        printf("value of A, A[%lu - 1]: %lu \n", i, A[i-1]);
        chunked_array[i] = chunked_array[i-1] + A[i-1];
        acc = chunked_array[i];
      }
      printf("Thread %d with offs %d.\n", omp_get_thread_num(), acc);

    
    // int offs = 0;
    // #pragma omp for schedule(static,chunk_size) reduction(+:acc)
    //   for (long i = 1; i < n; i++) {

    //     printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);

        

    //     prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    //     acc =  prefix_sum[i];
    //     offs = prefix_sum[i];
    // }

    // printf("Thread %d with offs %d.\n", omp_get_thread_num(), offs);

    // long this_thread = omp_get_thread_num();
    // shared[this_thread] = offs;

  }


  for (long j=0; j<size_of_shared_storage; j++) {
    printf("\t shared %lu: %lu \n", j, shared[j]);
  }




}

int main() {
  // long N = 100000000;
  long N = 24;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  //for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) A[i] = 1+ (rand() % 100);

  for (long i = 0; i < N; i++) B1[i] = 0;

  for (long i = 0; i < N; i++) B0[i] = 0;

  printf("A:\n");
  for (int x=0; x<N;x++) {
    printf(" %lu , ", A[x]);
  }
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  for (long i = 0; i < N; i++) {
    printf("B0[%lu], B1[i]-> %lu %lu \n", i, B0[i], B1[i]);

  }


  free(A);
  free(B0);
  free(B1);
  return 0;
}