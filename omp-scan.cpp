#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;

  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();

  if (n == 0) return;
  prefix_sum[0] = 0;

  int NTHREADS = 6;
  int size_of_shared_storage = NTHREADS;

  //  get the ceiling during division
  int chunk_size = 1 + ((n - 1) / NTHREADS);
  printf("N: %lu, chunk size: %i\n", n, chunk_size);

  // create a shared space for the threads to write offsets to
  long* shared = (long*) malloc(size_of_shared_storage * sizeof(long));
  for (long i = 0; i < size_of_shared_storage; i++) shared[i] = 0;

  // step 1. first pass, done in parallel
  long acc = 0;
  #pragma omp parallel num_threads(NTHREADS)
  {
    bool first = true;
    long offs = 0;
    #pragma omp for schedule(static,chunk_size)
      for (long i = 1; i < n; i++) {

        if (first == true) {

            // start with 0 as [prefix - 1]
            long start_value = 0;
            prefix_sum[i] = start_value + A[i-1];
            first = false;

        } else {
            prefix_sum[i] = prefix_sum[i-1] + A[i-1];
        }

        // take the partial sum as the saved value to be used for offset
        offs = prefix_sum[i];
      }
      int this_thread = omp_get_thread_num();
      // printf("Thread %d with offs %d.\n", omp_get_thread_num(), offs);
      // save the value in the shared array
      shared[this_thread] = offs;
  }

    // step 2. calculate the offset to add to each chunk
    long* offset_to_add = (long*) malloc(size_of_shared_storage * sizeof(long));
    for (long i = 0; i < size_of_shared_storage; i++) offset_to_add[i] = 0;
    
    for (int k=0;k<NTHREADS;k++) {
        long sum=0;

        for (int j=0;j<NTHREADS;j++) {
            if (j==k) break;
            sum = sum + shared[j];
        }
        offset_to_add[k] = sum;

    }
    // // check result, for debugging
    // for (long j=0; j<size_of_shared_storage; j++) {
    //     printf("\t final offset to add %lu: %lu \n", j, offset_to_add[j]);
    // }

    // step 3. next, add the offset in parallel
    #pragma omp parallel num_threads(NTHREADS)
    {
        #pragma omp for schedule(static,chunk_size) 
        for (long i = 1; i < n; i++) {

            int this_thread = omp_get_thread_num();
            // printf("Thread %d.\n", this_thread);

            prefix_sum[i] = prefix_sum[i] + offset_to_add[this_thread];
        }
    }
}

int main() {
  long N = 100000000;
  // long N = 100000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  for (long i = 0; i < N; i++) A[i] = rand();

  for (long i = 0; i < N; i++) B1[i] = 0;

  for (long i = 0; i < N; i++) B0[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

    // for debugging purposes
//   for (long i = 0; i < N; i++) {
//     printf("B0[%lu], B1[i]-> %lu %lu \n", i, B0[i], B1[i]);
//   }


  free(A);
  free(B0);
  free(B1);
  return 0;
}
