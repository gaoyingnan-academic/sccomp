functions{
 
  #include common_functions.stan
  #include singular_VCoV_matrix.stan
  
  real partial_sum_2_lpmf(
    // Parallel
    array[] int idx_y,
    int start,
    int end,
    
    // General
    int is_proportion,
    array[,] int y, //Sliced, M dimensions
    array[,] real y_proportion, // ILR, M-1 dimensions
    
    // Variance-Covariance
    array[] matrix L, // ILR, M-1 dimensions
    matrix intermediate_u, // Sliced, ILR
    
    // Fixed effects
    matrix beta, // ILR-transformed, M-1 dimensions 
    int M, 
    
    // Helmert matrix
    matrix Helmert
    ){
      
      int N = end-start+1; // Number of observations subsetted to this chunk
      
      // mu
      matrix[M-1, N] mu = (rep_matrix(1,N,1) * beta)';
 
      // target log-probability
      real target_lp = 0;
      
      // If input is proportions
      if(is_proportion){
        for(n in 1:N){
          target_lp += multi_normal_cholesky_lupdf(
            to_vector(y_proportion[idx_y[n]]) |
            mu[,n],
            to_matrix(L[1])
          );
        }
      }
      // If input is counts
      else{
        matrix[M,N] MU = Helmert*(mu + intermediate_u[idx_y,]'); // Back to CLR
        for(n in 1:N){
          target_lp += multinomial_logit_lupmf(
            to_array_1d(y[idx_y[n],]) | MU[,n]);
        }
      }
      return target_lp;
    }
}

data{
  // A minimal set of data and parameters for simplicity
  int<lower=0, upper=1> is_proportion;
  int<lower=1> N; // Equivalent to S in publication
  int<lower=1> M; // Equivalent to G in publication
  array[N] int exposure;
  array[N * !is_proportion,M] int<lower=0> y;
  array[N * is_proportion,M] real<lower=0, upper=1> y_proportion;

  // Verbose (for diagnostic and debugging)
  int<lower=0, upper=1> is_vb;
  
  // Prior info
  array[2] real prior_sd_intercept;
  array[2] real prior_mean_intercept;
  real<lower=0> prior_corr_eta; 
  
  // Exclude priors for testing purposes
  int<lower=0, upper=1> use_data;
  
  // Parallel chain
  int<lower=1> grainsize;
  
  // External constants
  
}

transformed data{
  // Internal constants
  matrix[M,M-1] Helmert = canonical_Helmert(M);
  
  // centered and isometric log-ratio transformed data (only relevant for proportional input)
  matrix[N * is_proportion,M] clr_y_proportion;
  matrix[N * is_proportion,M-1] ilr_y_proportion;
  
  // For multinomial acceleration by pseudo-vectorization
  array[N * !is_proportion] int ysum; // Supposedly the same as exposure in data block
  
  if(!is_proportion){
    for(n in 1:N) ysum[n] = sum(y[n,]); // But re-calculated here to enforce equivalence to the sum
  }else{
    clr_y_proportion = to_matrix(log(y_proportion));
    for(n in 1:N){
      clr_y_proportion[n] = clr_y_proportion[n] - mean(clr_y_proportion[n]);
    }
    ilr_y_proportion = (Helmert'*clr_y_proportion')';
  }
  
  // For CPU parallelisation
  array[N] int array_N;
  for(n in 1:N) array_N[n] = n;
  
}

parameters{
  // ILR means are not constrained
  matrix[M-1,1] beta_raw; // Only M-1 degree of freedom

  // ILR log-standard deviations, not constrained
  matrix[1, M-1] alpha_raw;
  
  // ILR Cholesky factors for correlation matrices
  array[1] cholesky_factor_corr[M-1] L; // Cholesky factor for correlation matrices
  
  // raw ILR latent residuals
  matrix[N* use_data, (M-1) * !is_proportion] intermediate_u_raw;

}

transformed parameters{
  // Non-centered parameterisation for ILR means
  matrix[M-1,1] beta = beta_raw*prior_mean_intercept[2] + prior_mean_intercept[1];
  
  // Non-centered parameterisation for ILR log-standard deviations
  matrix[1, M-1] alpha = alpha_raw*prior_sd_intercept[2] + prior_sd_intercept[1];
  
  // Non-centered parameterisation for ILR intermediate u
  array[1] cholesky_factor_cov[M-1] Lv;
  Lv[1] = diag_pre_multiply(exp(to_vector(alpha[1])),L[1]);
  matrix[N* use_data , (M-1) * !is_proportion] intermediate_u; // ILR
  for(n in 1:N){
      if(use_data&&(!is_proportion)){
        intermediate_u[n] = to_row_vector(
          Lv[1]*to_vector(intermediate_u_raw[n]));// Add multiplication by 2
    }
  }
  
}

model{
  // Fit main distribution
  if(use_data == 1){
    
   target += reduce_sum(
      partial_sum_2_lupmf,
      array_N,
      grainsize,
      
      // General
      is_proportion,
      y,
      to_array_2d(ilr_y_proportion),
      
      // Variance-Covariance
      Lv, // Only used when is_proportion
      intermediate_u, // Only used when !is_proportion

      // Fixed effects
      beta', 
      M, 
      
      // Helmert matrix
      Helmert
      );
  }
  alpha_raw[1]  ~ std_normal();
 
  // Priors abundance
  beta_raw[,1] ~ std_normal();

  // (Hyper-)priors for the correlation matrices
  L[1] ~ lkj_corr_cholesky(prior_corr_eta);
  
  // Priors for intermediate_u, only matters when using count data
  if(use_data&&(!is_proportion)){
    to_vector(intermediate_u_raw) ~ std_normal(); //student-t, cauchy, horseshoe
  }
  
}

generated quantities {
  // ILR means to CLR means for interpretability
  matrix[1,M] beta_CLR = (Helmert*beta)';

  // Return complete singular VCoV as standard deviations and correlation matrix
  matrix[M*is_vb, is_vb] sigma_CLR;
  matrix[M*is_vb, is_vb] alpha_CLR;
  array[is_vb] matrix[M,M] R_CLR;
  if(is_vb){
      (sigma_CLR[,1],R_CLR[1]) = get_std_and_corr_of_VCoV(
        Helmert*multiply_lower_tri_self_transpose(Lv[1])*Helmert'
        );
      alpha_CLR = log(sigma_CLR);
  }
  
}

