#include "driver.h"

int main(int argc, char *argv[])
{
  int
    m, n, k,
    ldA, ldB, ldC,
    size, first, last, inc,
    i, irep,
    nrepeats;

  double
    d_one = 1.0,
    dtime, dtime_best_ref = DP_MAX, dtime_best = DP_MAX, 
    diff, maxdiff = 0.0, gflops, per_of_ref;

  double
    *A, *B, *C, *Cold, *Cref;

  /* Every time trial is repeated "repeat" times and the fastest run in 
  recorded */
  nrepeats = 3;

  /* Timing trials for matrix sizes m=n=k=first to last in increments
     of inc will be performed.  (Actually, we are going to go from
     largest to smallest since this seems to give more reliable 
     timings.  */
  first = P_BEGIN;
  last  = P_END;
  inc   = P_INC;

  /* Adjust first and last so that they are multiples of inc */
  last = ( last / inc ) * inc;
  first = ( first / inc ) * inc;
  first = ( first == 0 ? inc : first );

  printf( "%d_group_data( 50, 1:5) = [ n      gflops   ref_gflops   %%_of_ref   diff       ];\n", GNUM );
  for ( size=last; size>= first; size-=inc ){

    /* we will only time cases where all three matrices are square */
    m = n = k = size;
    ldA = ldB = ldC = size;

    /* Gflops performed */
    gflops = 2.0 * m * n * k * 1e-09;

    /* 
      Allocate space for the matrices.  We will use five arrays:
      A will be the address where A is stored.   Addressed with alpha(i,j).
      B will be the address where B is stored.   Addressed with beta(i,j).
      C will be the address where C is stored.   Addressed with gamma(i,j).

      Now, we will compute C = A B + C with via routine MyGemm
      and also with a reference implementation.  Therefore, we will
      utilize two more arrays:

      Cold will be the address where the original matrix C is
      stored.  

      Cref will be the address where the result of computing C += AB
      computed with the reference implementation will be stored.
    */

    A = ( double * ) malloc( ldA * k * sizeof( double ) );
    B = ( double * ) malloc( ldB * n * sizeof( double ) );
    C = ( double * ) malloc( ldC * n * sizeof( double ) );
    Cold = ( double * ) malloc( ldC * n * sizeof( double ) );
    Cref = ( double * ) malloc( ldC * n * sizeof( double ) );

    /* Generate random matrix A */
    RandomMatrix( m, k, A, ldA );

    /* Generate random matrix B */
    RandomMatrix( k, n, B, ldB );

    /* Generate random matrix Cold */
    RandomMatrix( m, n, Cold, ldC );
    
    /* Time reference implementation provided by the BLAS library
       routine dgemm (double precision general matrix-matrix
       multiplicationn */
    for ( irep=0; irep<nrepeats; irep++ ){
      
      /* Copy matrix Cold to Cref */
      memcpy( Cref, Cold, ldC * n * sizeof( double ) );
    
      /* start clock */
      dtime = FLA_Clock();
    
      /* Compute Cref += A B*/
      dgemm_( "No transpose", "No transpose",
	      &m, &n, &k,
	      &d_one, A, &ldA,
	              B, &ldB,
	      &d_one, Cref, &ldC );

      /* stop clock */
      dtime = FLA_Clock() - dtime;

      /* store best run */
      dtime_best_ref = dmin(dtime, dtime_best_ref);
    }
  

    /* flush the output buffer because otherwise
      it may throw the timings of a next
      experiment. */
    fflush( stdout ); 

    /* Time MyGemm */
    for ( irep=0; irep<nrepeats; irep++ ){
      /* Copy vector Cold to C */
      memcpy( C, Cold, ldC * n * sizeof( double ) );
    
      /* start clock */
      dtime = FLA_Clock();
    
      /* Compute C += A B */
      MyGemm( m, n, k, A, ldA, B, ldB, C, ldC );

      /* stop clock */
      dtime = FLA_Clock() - dtime;
    
      /* store best run */
      dtime_best = dmin(dtime, dtime_best);
    }

    /* compute the diff of C and Cref and store max diff */
    diff = MaxAbsDiff( m, n, C, ldC, Cref, ldC );
    maxdiff = dmax(diff, maxdiff);

    /* compute GFLOPS / GFLOPS_ref */
    per_of_ref = dtime_best_ref/dtime_best;
    
    printf( "%1lu_group_data",(unsigned long) GNUM);
    printf( "( %2lu, 1:5) = [ %4lu %7.2f  %7.2f     %7.2f       %8.4le ];\n",
              (unsigned long) (size - first + 1)/inc + 1,
              (unsigned long) size, 
              gflops/dtime_best,
              gflops/dtime_best_ref,  
              per_of_ref,
              diff);

    /* flush the output buffer because otherwise
      it may throw the timings of a next
      experiment. */
    fflush( stdout );  

    /* Free the buffers */
    free( A );
    free( B );
    free( C );
    free( Cold );
    free( Cref );
  }
  printf( "\n\n%% Maximum difference between reference and your implementation: %le.\n", maxdiff );
  
  exit( 0 );
}
