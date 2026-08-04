functions{

  #include singular_VCoV_matrix.stan
  #include matrix_exponential_and_logarithm.stan

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
    array[,] int y, //Sliced, M dimensions
    matrix y_proportion, // Sliced, ILR, M-1 dimensions
    
    // Fixed effects
    vector Xalpha_shift,
    matrix Xbeta, // ILR-transformed, M-1 dimensions 
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
      
      // If input is proportions
      if(is_proportion){
        for(n in 1:N){
          target_lp += multi_normal_cholesky_lpdf(
            y_proportion[n,] |
            to_vector(Xbeta[Xn_to_XQ[idx_y[n],2],]), 
            exp(Xalpha_shift[Xn_to_XQ[idx_y[n],1]]) * XLv[Xn_to_XQ[idx_y[n],3]]
          );
        }
      }
      // If input is counts
      else{
        for(n in 1:N){
          target_lp += laplace_marginal_tol(
            ll_function, //custom log-likelihood function with latent residuals
            (M,to_array_1d(y[idx_y[n],]),to_vector(Xbeta[Xn_to_XQ[idx_y[n],2],]),Helmert), //parameters without latent residuals
            M-1, // Hessian block size
            cov_function, //covariance function for latent residuals
            (exp(Xalpha_shift[Xn_to_XQ[idx_y[n],1]]) * XLv[Xn_to_XQ[idx_y[n],3]],M), // cov_function just converts it to VCoV
            laplace_ops // options for laplace approximation
            );
        }
      }
      return target_lp;
    }
}

data{
  // Input data type and dimensions
  int<lower=0, upper=1> is_proportion;
  int<lower=1> N; // Equivalent to S in publication
  int<lower=3> M; // Equivalent to G in publication
  array[N * !is_proportion,M] int<lower=0> y; // Counts
  array[N * is_proportion,M] real<lower=0, upper=1> y_proportion; // Proportions

  // Linear design
  int<lower=1> A; // Number of columns in variability scaling (alpha-shift) design
  int<lower=1> B; // Number of columns in mean (beta) design
  int<lower=1> C; // Number of columns in covariances design

  int<lower=1> A_intercept_columns; // Number of intercept columns in variability scaling design
  int<lower=1> B_intercept_columns; // Number of intercept columns in mean (beta) design
  int<lower=1> C_intercept_columns; // Number of intercept columns in covariances design
  
  int<lower=1> Ar; // Number of unique rows in variability scaling design
  int<lower=1> Br; // Number of unique rows in mean (beta) design
  int<lower=1> Cr; // Number of unique rows in covariances design
  
  matrix[Ar, A] XA; // The unique variability scaling design
  matrix[Br, B] XB; // The unique mean design
  matrix[Cr, C] XC; // The unique covariance design
  array[N,3] int Xn_to_XQ; // The indice map of sample designs to unique designs

  // Verbose (for diagnostic and debugging)
  int<lower=0, upper=1> is_vb;
  
  // Prior info
  array[2] real prior_scale_norm_mean;
  array[2] real prior_scale_norm_sd;
  array[2] real prior_mean_norm_sd; // mean of normal prior must be 0 for means
  array[2] real prior_sd_norm_sd; // mean of normal prior must be 0 for normalised sd
  array[2] real<lower=0> prior_corr_lkj_eta; 

  // Random effect designs


  // Priors for random effects
  
  
  // Exclude data for testing purposes
  int<lower=0, upper=1> use_data;
  
  // Parallel chain
  int<lower=1> grainsize;
  
  // Leave-one-out cross-validation
  int<lower=0, upper=1> enable_loo;
  
  // Simulation-based goodness-of-fit assessment
  int<lower=0, upper=1> enable_gof;
  
  // options for Laplace approximation
  real<lower=0> laplace_tol; // tolerance for optimizer
  int<lower=0> laplace_max_iter; // maximum number of steps for optimizer
  int<lower=1, upper=3> laplace_solver; // Newton solver type being used
  
  // External constants
  int<lower=0, upper=1> use_external_basis; // Allow user-supplied isometric basis
  matrix[M,M-1] external_basis; 
}

transformed data{
  // Internal constants
  int Cholesky_df = ((M-1)*(M-2))%/%2; // Avoid int divisions when declaring vars
  matrix[M,M-1] Helmert;
  if(use_external_basis){
    Helmert = external_basis; // The external basis is not checked inside Stan
  }else{
    Helmert = canonical_Helmert(M);
  }
  matrix[M-1,M-2] H_alpha = canonical_Helmert(M-1);
  
  // centered and isometric log-ratio transformed data (only relevant for proportional input)
  matrix[N * is_proportion,M] clr_y_proportion;
  matrix[N * is_proportion,M-1] ilr_y_proportion;
  
  // total count per sample
  array[N * !is_proportion] int ysum; // used for simulation in generated quantities
  
  // initialise transformed/derived data
  if(!is_proportion){
    for(n in 1:N) ysum[n] = sum(y[n,]);
  }else{
    for(n in 1:N){
      clr_y_proportion[n] = log(to_row_vector(y_proportion[n,]));
      clr_y_proportion[n] = clr_y_proportion[n] - mean(clr_y_proportion[n]);
    }
    ilr_y_proportion = clr_y_proportion*Helmert;
  }
  
  // For CPU parallelisation
  array[N] int array_N;
  for(n in 1:N) array_N[n] = n;
  
  // For Laplace approximation
  tuple(vector[M-1], real, int, int, int, int) laplace_ops = generate_laplace_options(M-1);
  laplace_ops.2 = laplace_tol;      // tolerance for optimizer
  laplace_ops.3 = laplace_max_iter; // maximum number of steps for optimizer
  laplace_ops.4 = laplace_solver;   // solver type being used
  
}

parameters{
  // Scaling of variability, not constrained
  vector[A] alpha_shift_raw;
  
  // ILR means, not constrained
  matrix[B, M-1] beta_raw; // M-1 degree of freedom

  // ILR log-standard deviations, normalised
  matrix[C, M-2] alpha_raw; // M-2 degree of freedom as scale moves out
  
  // ILR Cholesky factors for correlation matrices
  array[C] cholesky_factor_corr[M-1] L;
  
  //
}

transformed parameters{
  // Non-centered parameterisation for variability scales
  vector[A] alpha_shift = alpha_shift_raw * prior_scale_norm_sd[2] + prior_scale_norm_mean[2];
  alpha_shift[1:A_intercept_columns] = 
    alpha_shift_raw[1:A_intercept_columns] * prior_scale_norm_sd[1] + 
    prior_scale_norm_mean[1];
  
  // Non-centered parameterisation for ILR means
  // Note that mean of the prior distribution of means must be 0
  matrix[B, M-1] beta = beta_raw * prior_mean_norm_sd[2];
  beta[1:B_intercept_columns,] = beta_raw[1:B_intercept_columns,] * prior_mean_norm_sd[1];
  
  // Non-centered parameterisation for ILR normalised log-standard deviations
  matrix[C, M-1] alpha = alpha_raw * H_alpha' * prior_sd_norm_sd[2];
  alpha[1:C_intercept_columns,] = 
    alpha_raw[1:C_intercept_columns,] * H_alpha' * prior_sd_norm_sd[1];
  // normalise the generalised variance (determinant of VCoV) to 1 and take matrix log
  matrix[C,(M-1)*(M-1)] tV; // vectorised matrix logarithm of VCoV
  for(c in 1:C){
    alpha[c,] = alpha[c,] - sum(log(diagonal(L[c])))/(M-1);
    tV[c,] = to_row_vector(vectorized_matrix_log_spd(
      multiply_lower_tri_self_transpose(diag_pre_multiply(alpha[c,],L[c]))
    ));
  }

  // apply designs for parameters
  vector[Ar] Xalpha_shift = XA*alpha_shift;
  matrix[Br, M-1] Xbeta = XB*beta;
  matrix[Cr, (M-1)*(M-1)] XtV = XC*tV;
  array[Cr] matrix[M-1,M-1] XLv;
  for(cr in 1:Cr){
    XLv[cr] = unvectorized_matrix_exp_spd(to_vector(XtV[cr,]));
  }
  
}

model{
  // Fit main distribution
  if(use_data == 1){
    
   target += reduce_sum(
      partial_sum_2_lupmf,
      array_N,
      grainsize,
      
      // Data
      M,
      is_proportion,
      y,
      ilr_y_proportion,
      
      // Fixed effects
      Xalpha_shift,
      Xbeta,
      XLv,
      
      // Fixed effect mapping
      Xn_to_XQ,
      
      // Helmert matrix
      Helmert,
      
      // Laplace approximation options
      laplace_ops
      );
  }
  
  // Priors
  alpha_shift_raw  ~ std_normal();
  for(b in 1:B) beta_raw[b,] ~ std_normal();
  for(c in 1:C_intercept_columns){
    alpha_raw[c,] ~ std_normal();
    L[c] ~ lkj_corr_cholesky(prior_corr_lkj_eta[1]);
  }
  if(C_intercept_columns < C){
    for(c in (C_intercept_columns+1):C){
      alpha_raw[c,] ~ std_normal();
      L[c] ~ lkj_corr_cholesky(prior_corr_lkj_eta[2]);
    }
  }
  
}

generated quantities {
  // Sanity check for sampled parameters
  vector[Cr*is_vb] log_detV;
  if(is_vb){
    for(c in 1:Cr)
    log_detV[c] = sum(log(diagonal(XLv[c]))); // This should always be 0
  }
  
  // Transform parameters back to CLR space for interpretability
  // Scaling remains the same and does not need transformation
  matrix[B,M] beta_CLR = beta*Helmert';
  matrix[Br,M] Xbeta_CLR = Xbeta*Helmert';

  // Covariances are interperted after designs are applied
  matrix[Cr, M] sigma_CLR;
  matrix[Cr, M] alpha_CLR;
  array[Cr] matrix[M,M] R_CLR;
  for(c in 1:Cr){
    (sigma_CLR[c,],R_CLR[c]) = get_std_and_corr_of_VCoV(
      Helmert*multiply_lower_tri_self_transpose(XLv[c])*Helmert');
    alpha_CLR[c,] = log(sigma_CLR[c,]);
  }
  
  // Goodness-of-fit through simulation
  matrix[N*enable_gof, M] sim_mu;
  array[N*enable_gof * !is_proportion, M] int sim_y;
  if(enable_gof==1){
    matrix[M-1,N] latent_mu;
    for(n in 1:N){
      latent_mu[,n] = multi_normal_cholesky_rng(
        to_vector(Xbeta[Xn_to_XQ[n,2],]), 
        exp(Xalpha_shift[Xn_to_XQ[n,1]]) * XLv[Xn_to_XQ[n,3]]);
      sim_mu[n,] = to_row_vector(Helmert*latent_mu[,n]);
      }
    if(!is_proportion){
      for(n in 1:N) sim_y[n,] = multinomial_logit_rng(to_vector(sim_mu[n,]), ysum[n]);
    }
  }
  
  // LOO
  vector[N*enable_loo] log_lik;
  if(enable_loo==1){
    // reuse partial sum function to sync any change in LL calculation
    for(n in 1:N){
      log_lik[n] = partial_sum_2_lpmf(
        {n}|n,n, // per-sample block
        // Data
        M, is_proportion, y, ilr_y_proportion,
        // Fixed effects
        Xalpha_shift, Xbeta, XLv,
        // Fixed effect mapping
        Xn_to_XQ,
        // Helmert matrix
        Helmert,
        // Laplace approximation options
        laplace_ops
      );
    }
  }
}
