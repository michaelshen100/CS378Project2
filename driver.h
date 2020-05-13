#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

////////////////////////////////////////////////////////////////////////////////
/* Macros */

#define DP_MAX 99999999.0

// computes absolute value
#define dabs(x) ( (x) < 0 ? -(x) : x )

// returns min value
#define dmin(x, y) ( x < y ? x : y )

// returns max value
#define dmax(x, y) ( x > y ? x : y )

/* Get time function borrowed from libflame library */
double FLA_Clock();

////////////////////////////////////////////////////////////////////////////////
/* Function prototypes */

/* Compute the maximum absolute difference over all
  corresponding elements of two matrices */
double MaxAbsDiff( int, int, double *, int, double *, int );

/* RandomMatrix overwrites a matrix with random values */
void RandomMatrix( int, int, double *, int );

/* Prototype for BLAS matrix-matrix multiplication routine (which we will 
   use for the reference implementation */
void dgemm_(  char *, char *,                 // transA, transB
	            int *, int *, int *,            // m, n, k
	            double *, double *, int *,      // alpha, A, ldA
	            double *, int *,                // B, ldB
	            double *, double *, int *       // beta, C, ldC
            );  

/* My_Gemm is a common interface to all the implementations we will 
   develop so we don't have to keep rewriting this driver routine. */
void MyGemm( int, int, int, double *, int, double *, int, double *, int );

////////////////////////////////////////////////////////////////////////////////