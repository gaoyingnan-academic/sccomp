// canonical Helmert matrix
matrix canonical_Helmert(int M){
  matrix[M,M-1] H = rep_matrix(0,M,M-1);
  for(m in 1:(M-1)){
    H[1:m,m] = rep_vector(1/sqrt(m*(m+1)),m);
    H[m+1,m] = -m/sqrt(m*(m+1));
  }
  return H;
}

// likelihood of data given latent residual + parameters:
real ll_function(
    vector u, // latent ILR residuals
    int M, // number of taxa for convenience
    array[] int y, // observed counts
    vector beta, // ILR mean
    matrix Helmert // canonical Helmert matrix constant
) {
  real lp = 0;
  vector[M] mu = Helmert*(beta+u);
  lp += multinomial_logit_lpmf(
    y | mu
  );
  return lp;
}

// multinormal distribution of latent residuals
matrix cov_function(
  matrix L,
  int M
){
  return multiply_lower_tri_self_transpose(L); // Simply transform L to VCoV
}

// partial likelihood for CPU parallelisation
real partial_sum_2_lpmf(
  // Parallel
  array[] int idx_y,
  int start,
  int end,
  
  // Data
  int M, 
  int is_proportion,
  array[,] int y, // Sliced, M dimensions
  matrix y_proportion, // Unsliced, ILR, M-1 dimensions
  
  // Fixed effects
  vector Xalpha_shift,
  matrix Xbeta, // ILR-transformed, M-1 dimensions 
  matrix XtV, // design-specific logVCov, M-1 dimensions
  int precalculate_vcov, // whether XLv supplied below has been fully initialised
  int taylor_K, // terms to use in Taylor's expansion
  int taylor_s, // scaling of exponential so Taylor's expansion
  array[] matrix XLv, // Cholesky factor of covariances, ILR, M-1 dimensions
  
  // Fixed effect mapping
  array[,] int Xn_to_XQ,
  
  // Helmert matrix
  matrix Helmert,
  
  // Laplace approximation options
  data tuple(vector, real, int, int, int, int) laplace_ops
  ){
    int N = end-start+1; // Number of observations subsetted to this chunk
    // target log-probability
    real target_lp = 0;
    
    // calculate XLv when not precalculated
    matrix[M-1,M-1] this_XLv;

    // If input is proportions
    if(is_proportion){
      for(n in 1:N){
        if(precalculate_vcov){
          this_XLv = XLv[Xn_to_XQ[idx_y[n],3]];
        }
        else{
          // expm via truncated Taylor's series
          this_XLv = unvectorized_matrix_exp_taylor(to_vector(XtV[Xn_to_XQ[idx_y[n],3],]),taylor_K,taylor_s);
        }
        target_lp += multi_normal_cholesky_lpdf(
          y_proportion[idx_y[n],] |
          to_vector(Xbeta[Xn_to_XQ[idx_y[n],2],]), 
          exp(Xalpha_shift[Xn_to_XQ[idx_y[n],1]]) * this_XLv
        );
      }
    }
    // If input is counts
    else{
      for(n in 1:N){
        if(precalculate_vcov){
          this_XLv = XLv[Xn_to_XQ[idx_y[n],3]];
        }
        else{
          // expm via truncated Taylor's series
          this_XLv = unvectorized_matrix_exp_taylor(to_vector(XtV[Xn_to_XQ[idx_y[n],3],]),taylor_K,taylor_s);
        }
        target_lp += laplace_marginal_tol(
          ll_function, //custom log-likelihood function with latent residuals
          (M,to_array_1d(y[idx_y[n],]),to_vector(Xbeta[Xn_to_XQ[idx_y[n],2],]),Helmert), //parameters without latent residuals
          M-1, // Hessian block size
          cov_function, //covariance function for latent residuals
          (exp(Xalpha_shift[Xn_to_XQ[idx_y[n],1]]) * this_XLv,M), // cov_function just converts it to VCoV
          laplace_ops // options for laplace approximation
          );
      }
    }
    return target_lp;
  }