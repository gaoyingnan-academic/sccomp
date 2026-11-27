functions{
  #include common_functions.stan
  #include LNM_under_ILR.stan
  #include matrix_exponential_and_logarithm.stan
  // To ensure that sampler and quantity generator use the same functions
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
  // Sample → unique-row map: cols 1–3 = A/B/C FE; cols 4–5 = composition RE1 / RE2
  array[N,5] int Xn_to_XQ;

  // Verbose (for diagnostic and debugging)
  int<lower=0, upper=1> is_vb;
  
  // Prior info
  vector[A] prior_scale_norm_mean;
  vector<lower=0>[A]  prior_scale_norm_sd;
  matrix[B, M-1] prior_mean_norm_mean;
  matrix<lower=0>[B, M-1]  prior_mean_norm_sd;
  matrix[C, M-2] prior_diagV_norm_mean;
  matrix<lower=0>[C, M-2]  prior_diagV_norm_sd;
  matrix[C, ((M-1)*(M-2))%/%2] prior_ltriV_norm_mean;
  matrix<lower=0>[C, ((M-1)*(M-2))%/%2] prior_ltriV_norm_sd;
  
  // Composition random effects on the B (mean) axis — max 2 groupings.
  // Unique-row designs XBR / XBR2 (like XA/XB/XC); gate on ncol_XBR[k] > 0 (no is_random_effect).
  // ILR-native coeffs (M-1); no mean–variability association.
  array[2] int ncol_XBR;
  int<lower=0> BRr;  // unique rows in XBR
  int<lower=0> BRr2; // unique rows in XBR2
  matrix[BRr, ncol_XBR[1]] XBR;
  matrix[BRr2, ncol_XBR[2]] XBR2;

  // Covariance setup within each RE grouping (per ILR coordinate)
  array[2] int n_groups_B;
  array[2] int how_many_factors_in_XBR;
  array[how_many_factors_in_XBR[1], n_groups_B[1]] int group_factor_indexes_XBR;
  array[how_many_factors_in_XBR[2], n_groups_B[2]] int group_factor_indexes_XBR2;
  
  // Pre-calculate design-specific VCoV
  int<lower=0, upper=1> precalculate_vcov;
  
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
  
  // options for matrix exponential approximation via Taylor's expansion
  int<lower=1> taylor_K; // terms to use in Taylor's expansion
  int<lower=1> taylor_s; // scaling of exponential so Taylor's expansion
  
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
  matrix[M-1,M-2] H_minus_1 = canonical_Helmert(M-1);
  
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

  // ILR log-VCoV diagonal elements, normalised
  matrix[C, M-2] diag_tV_raw; // M-2 degree of freedom as scale moves out
  
  // ILR log-VCoV lower-triangular elements, unconstrained
  matrix[C, Cholesky_df] offdiag_tV_raw;

  // Composition RE: unconstrained ILR (M-1) per RE column (B axis)
  array[ncol_XBR[1]] vector[M-1] beta_random_raw;
  array[ncol_XBR[2]] vector[M-1] beta_random_raw_2;

  // SD hyperparameters of random effects (slot [1]/[2] for the two groupings)
  array[2 * (ncol_XBR[1]>0)] real beta_random_sigma_mu;
  array[2 * (ncol_XBR[1]>0)] real beta_random_sigma_sigma;

  // Within-grouping factor covariance (per ILR coordinate)
  array[(M-1) * (ncol_XBR[1]>0)] vector[how_many_factors_in_XBR[1]] beta_random_sigma_raw;
  array[(M-1) * (ncol_XBR[1]>0)] cholesky_factor_corr[how_many_factors_in_XBR[1] * (ncol_XBR[1]>0)] sigma_correlation_factor_B;

  array[(M-1) * (ncol_XBR[2]>0)] vector[how_many_factors_in_XBR[2]] beta_random_sigma_raw_2;
  array[(M-1) * (ncol_XBR[2]>0)] cholesky_factor_corr[how_many_factors_in_XBR[2] * (ncol_XBR[2]>0)] sigma_correlation_factor_B_2;

  // If I have just one group
  array[ncol_XBR[1]>0] real zero_beta_random;
}

transformed parameters{
  // Non-centered parameterisation for variability scales
  vector[A] alpha_shift = alpha_shift_raw .* prior_scale_norm_sd + prior_scale_norm_mean;
  
  // Non-centered parameterisation for ILR means
  // Note that mean of the prior distribution of means must be 0
  matrix[B, M-1] beta = beta_raw .* prior_mean_norm_sd + prior_mean_norm_mean;

  // Non-centered parameterisation for ILR normalised log-VCoV
  // constraining determinant of VCoV to 1 equivalent to sum-to-zero on diagonal elements of log-VCoV
  matrix[C, M-2] diag_tV_raw_scaled = diag_tV_raw .* prior_diagV_norm_sd + prior_diagV_norm_mean;
  matrix[C, M-1] diag_tV = diag_tV_raw_scaled * H_minus_1';

  // off-diagonal elements in log VCoV is unconstrained at all
  matrix[C, Cholesky_df] offdiag_tV = offdiag_tV_raw .* prior_ltriV_norm_sd + prior_ltriV_norm_mean;
    
  // assemble diagonal and off-diagonal elements in reshaped 1D log-VCoV
  matrix[C,(M-1)*(M-1)] tV;
  for(c in 1:C){
    tV[c,] = to_row_vector(
      sym_matrix_from_vectors(to_vector(diag_tV[c,]), to_vector(offdiag_tV[c,]))
      );
  }

  // apply designs for parameters
  vector[Ar] Xalpha_shift = XA*alpha_shift;
  matrix[Br, M-1] Xbeta = XB*beta;
  matrix[Cr, (M-1)*(M-1)] XtV = XC*tV;

  // Non-centered SD of random effects (per ILR coordinate)
  array[(M-1) * (ncol_XBR[1]>0)] vector[how_many_factors_in_XBR[1]] beta_random_sigma;
  if(ncol_XBR[1]>0) for(k in 1:(M-1)) beta_random_sigma[k] = beta_random_sigma_mu[1] + beta_random_sigma_sigma[1] * beta_random_sigma_raw[k];
  if(ncol_XBR[1]>0) for(k in 1:(M-1)) beta_random_sigma[k] = exp(beta_random_sigma[k]/3.0);

  array[(M-1) * (ncol_XBR[2]>0)] vector[how_many_factors_in_XBR[2]] beta_random_sigma_2;
  if(ncol_XBR[2]>0) for(k in 1:(M-1)) beta_random_sigma_2[k] = beta_random_sigma_mu[2] + beta_random_sigma_sigma[2] * beta_random_sigma_raw_2[k];
  if(ncol_XBR[2]>0) for(k in 1:(M-1)) beta_random_sigma_2[k] = exp(beta_random_sigma_2[k]/3.0);

  matrix[ncol_XBR[1], M-1] beta_random;
  matrix[ncol_XBR[2], M-1] beta_random_2;
  // Unique-row RE contributions (looked up via Xn_to_XQ[,4]/[,5])
  matrix[BRr, M-1] XBRbeta;
  matrix[BRr2, M-1] XBRbeta2;

  if(ncol_XBR[1]>0){
    beta_random =
      get_random_effect_matrix(
        M-1,
        n_groups_B[1],
        how_many_factors_in_XBR[1],
        1, // unused inside get_random_effect_matrix; gated by ncol_XBR
        ncol_XBR[1],
        group_factor_indexes_XBR,
        beta_random_raw,
        beta_random_sigma,
        sigma_correlation_factor_B
      );
    XBRbeta = XBR * beta_random;
  }

  if(ncol_XBR[2]>0){
    beta_random_2 =
      get_random_effect_matrix(
        M-1,
        n_groups_B[2],
        how_many_factors_in_XBR[2],
        1,
        ncol_XBR[2],
        group_factor_indexes_XBR2,
        beta_random_raw_2,
        beta_random_sigma_2,
        sigma_correlation_factor_B_2
      );
    XBRbeta2 = XBR2 * beta_random_2;
  }
}

model{
  // Cholesky factor corresponding to log-VCoV is computationally expensive
  array[Cr*precalculate_vcov] matrix[M-1,M-1] XLv = 
    rep_array(diag_matrix(rep_vector(1.0,M-1)),Cr*precalculate_vcov);
  if(precalculate_vcov){
    for(cr in 1:Cr){
    // expm via truncated Taylor's series
    XLv[cr] = unvectorized_matrix_exp_taylor(to_vector(XtV[cr,]),taylor_K,taylor_s);
    }
  }
  
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
      XtV,
      precalculate_vcov, 
      taylor_K, 
      taylor_s,
      XLv,
      
      // Unique-row map (FE + RE)
      Xn_to_XQ,

      // Composition RE (unique-row ILR contributions)
      ncol_XBR,
      XBRbeta,
      XBRbeta2,
      
      // Helmert matrix
      Helmert,
      
      // Laplace approximation options
      laplace_ops
      );
  }
  
  // Priors
  alpha_shift_raw  ~ std_normal();
  for(b in 1:B) beta_raw[b,] ~ std_normal();
  for(c in 1:C){
    diag_tV_raw[c,] ~ std_normal();
    offdiag_tV_raw[c] ~ std_normal();
  }

  // Composition RE priors (ILR-native; no mean–variability association)
  if(ncol_XBR[1]>0){
    for(k in 1:(M-1)) beta_random_raw[,k] ~ std_normal();
    for(k in 1:(M-1)) beta_random_sigma_raw[k] ~ std_normal();
    for(k in 1:(M-1)) sigma_correlation_factor_B[k] ~ lkj_corr_cholesky(2);

    beta_random_sigma_mu ~ std_normal();
    beta_random_sigma_sigma ~ std_normal();

    zero_beta_random ~ std_normal();
  }
  if(ncol_XBR[2]>0){
    for(k in 1:(M-1)) beta_random_raw_2[,k] ~ std_normal();
    for(k in 1:(M-1)) beta_random_sigma_raw_2[k] ~ std_normal();
    for(k in 1:(M-1)) sigma_correlation_factor_B_2[k] ~ lkj_corr_cholesky(2);
  }
}
