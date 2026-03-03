functions{
 
  #include common_functions.stan
  #include transform_cholesky_factor.stan
  #include singular_VCoV_matrix.stan
  
  // No bimodality for Wishart-based mean-variability association
  matrix abundance_variability_regression(row_vector abundance, array[] real prec_coeff, matrix I_J, int use_intercept){
    int M = num_elements(abundance);
    vector[M] variability =  abundance' * prec_coeff[2] + use_intercept*prec_coeff[1]; // as log-std
    matrix[M,M] S = I_J*diag_matrix(M*exp(2*variability)/(M-1))*I_J;
    return cholesky_decompose(S[1:(M-1),1:(M-1)]);
  }
      
  real partial_sum_2_lpmf(
    // Parallel
    array[] int idx_y,
    int start,
    int end,
    
    // General
    int is_proportion,
    array[,] int y,
    array[,] real y_proportion,
    array[] int ysum, //Sliced
    
    // Variance-Covariance
    array[] matrix XSigma, 
    matrix intermediate_u, // Sliced
    array[] int Xa_to_XA, // Sliced
    
    // Fixed effects
    matrix X,                   // Sliced
    matrix beta, 
    int M, 
    
    // Random effects
    array[] int ncol_X_random_eff,
    matrix X_random_effect,   // Sliced
    matrix X_random_effect_2,  // Sliced
    matrix random_effect,
    matrix random_effect_2
    
    ){
      
      int N = end-start+1; // Number of observations subsetted to this chunk
      
      // mu
      matrix[M, N] mu = (X[idx_y,] * beta)';
      
      if(ncol_X_random_eff[1]> 0)
      mu = mu + (X_random_effect[idx_y,] * random_effect)';
      
      if(ncol_X_random_eff[2]>0 )
      mu = mu + (X_random_effect_2[idx_y,] * random_effect_2)';

      // target log-probability
      real target_lp = 0;
      
      // If input is proportions
      if(is_proportion){
        for(n in 1:N){
          target_lp += multi_normal_cholesky_lupdf(
            to_vector(y_proportion[idx_y[n],1:(M-1)]) |
            mu[1:(M-1),idx_y[n]],
            to_matrix(XSigma[Xa_to_XA[idx_y[n]]])
          );
        }
      }
      // If input is counts
      else{
        mu = mu + intermediate_u[idx_y,]';
        for(n in 1:N){
          mu[,n] = softmax(mu[,n]);
        }
        // GPU-compatible pseudo-vectorization
        target_lp += poisson_lupmf(
          to_array_1d(y[idx_y,]) |
          to_vector(diag_post_multiply(mu,to_vector(ysum[idx_y])))
        ) - poisson_lupmf(
          ysum[idx_y] |
          to_vector(ysum[idx_y])
        );
      }
      return target_lp;
    }

}

data{
  int<lower=0, upper=1> is_proportion;
  int<lower=1> N; // Equivalent to S in publication
  int<lower=1> M; // Equivalent to G in publication
  int<lower=1> C;
  int<lower=1> A; // How many column in variability design
  int<lower=1> A_intercept_columns; // How many intercept column in varibility design
  int<lower=1> B_intercept_columns; // How many intercept column in varibility design
  int<lower=1> Ar; // Rows of unique variability design
  array[N] int exposure;
  array[N * !is_proportion,M] int<lower=0> y;
  array[N * is_proportion,M] real<lower=0, upper=1> y_proportion;
  matrix[N, C] X;
  matrix[Ar, A] XA; // The unique variability and correlation design
  matrix[N, A] Xa; // The variability and correlation design
  array[N] int Xa_to_XA; // The indice map of Xa rows to XA rows
  
  // Truncation (not used but kept for compatibility)
  int is_truncated;
  array[N,M] int truncation_up;
  array[N,M] int truncation_down;
  int<lower=1, upper=N*M> TNS; // truncation_not_size
  array[TNS] int<lower=1, upper=N*M> truncation_not_idx;
  int TNIM; // truncation_not_size
  array[TNIM,2] int<lower=1, upper=N*M> truncation_not_idx_minimal;
  
  // Verbose (for diagnostic and debugging)
  int<lower=0, upper=1> is_vb;
  //int<lower=0, upper=1> return_sample_params; // return unique sample-specific parameters
  
  // Prior info
  array[2] real prior_prec_intercept;
  array[2] real prior_prec_slope;
  array[2] real prior_prec_sd; // only [1](shape) matters by controling how nu varies for Wishart distribution
  array[2] real prior_mean_intercept; // Mean [1] will be ignored
  array[2] real prior_mean_coefficients; // Mean [1] will be ignored
  real<lower=0> prior_corr_eta; // controls the mean of nu for Wishart distribution
  
  // Exclude priors for testing purposes
  int<lower=0, upper=1> exclude_priors;
  int<lower=0, upper=1> bimodal_mean_variability_association;
  int<lower=0, upper=1> use_data;
  
  // Parallel chain
  int<lower=1> grainsize;
  
  // Does the design includes intercept
  int <lower=0, upper=1> intercept_in_design;
  
  // Random intercept
  
  // Is the parameters in random effect matrix, minus ther sub to zero parameters, for example if I have four groups, this will be 3
  int is_random_effect;
  
  // Is the parameters in random effect matrix
  array[2] int ncol_X_random_eff;
  matrix[N, ncol_X_random_eff[1]] X_random_effect;
  matrix[N, ncol_X_random_eff[2]] X_random_effect_2;
  
  // Covariance setup
  array[2] int n_groups;
  array[2] int how_many_factors_in_random_design;
  array[how_many_factors_in_random_design[1], n_groups[1]] int group_factor_indexes_for_covariance;
  array[how_many_factors_in_random_design[2], n_groups[2]] int group_factor_indexes_for_covariance_2;
  
  // LOO
  int<lower=0, upper=1> enable_loo;
  
  // External constants
  matrix[M,M-1] Helmert; // not really used here but kept for compatibility with the ILR solution
}

transformed data{
  // For multinomial acceleration by pseudo-vectorization
  array[N * !is_proportion] int ysum; // Supposedly the same as exposure in data block
  if(!is_proportion){
    for(n in 1:N) ysum[n] = sum(y[n,]); // But re-calculated here to enforce equivalence to the sum
  }
  
  // EXCEPTION MADE FOR WINDOWS GENERATE QUANTITIES IF RANDOM EFFECT DO NOT EXIST
  int ncol_X_random_eff_WINDOWS_BUG_FIX = max(ncol_X_random_eff[1], 1);
  int ncol_X_random_eff_WINDOWS_BUG_FIX_2 = max(ncol_X_random_eff[2], 1);
  
  // For parallelisation
  array[N] int array_N;
  for(n in 1:N) array_N[n] = n;
  
  // For proportional data
  matrix[N * is_proportion,M] y_proportion_clr_transformed;
  if(is_proportion){
    y_proportion_clr_transformed = to_matrix(log(y_proportion));
    for(n in 1:N){
      y_proportion_clr_transformed[n] = y_proportion_clr_transformed[n] - mean(y_proportion_clr_transformed[n]);
    }
  }
  
  // Internal constants
  cholesky_factor_cov[M-1,M-1] S_0 = cholesky_decompose(equal_variance_sum_to_zero_VCoV(M-1,0));
  vector[(M*(M-1))%/%2] vec_S_0 = transform_cholesky_factor_cov(S_0);
  matrix[M,M] I_J = I_minus_J_matrix(M);
  real nu_lower = M+1; // enforce Wishart distribution to be unimodal with location S and precision nu
}

parameters{
  // Use built-in sum-to-zero vector
  array[C] sum_to_zero_vector[M] beta_raw; // Each row is a sum_to_zero_vector of length M
  
  // Covariance matrix is singular so we only declare the full-rank [M-1] part
  //array[A] cov_matrix[M-1] Sigma_raw;
  array[A] cholesky_factor_cov[M-1] Sigma_raw;
  
  // full-rank part of residuals to bridge proportions and read counts
  matrix[N * !is_proportion, M-1] intermediate_u_raw; 
  
  // To exclude
  array[2] real prec_coeff;
  real<lower=0> prec_sd;

  // Random intercept // array of sum_to_zero_vector for each random effect
  array[ncol_X_random_eff[1] * (is_random_effect>0)] sum_to_zero_vector[M] random_effect_raw;
  array[ncol_X_random_eff[2] * (ncol_X_random_eff[2]>0)] sum_to_zero_vector[M] random_effect_raw_2;
  
  // sd of random intercept
  array[2 * (is_random_effect>0)] real random_effect_sigma_mu;
  array[2 * (is_random_effect>0)] real random_effect_sigma_sigma;
  
  // Covariance
  array[M * (is_random_effect>0)] vector[how_many_factors_in_random_design[1]]  random_effect_sigma_raw;
  array[M * (is_random_effect>0)] cholesky_factor_corr[how_many_factors_in_random_design[1] * (is_random_effect>0)] sigma_correlation_factor;
  
  // Covariance
  array[M * (is_random_effect>0)] vector[how_many_factors_in_random_design[2]]  random_effect_sigma_raw_2;
  array[M * (is_random_effect>0)] cholesky_factor_corr[how_many_factors_in_random_design[2] * (is_random_effect>0)] sigma_correlation_factor_2;
  
  // If I have just one group
  array[is_random_effect>0] real zero_random_effect;
  
}

transformed parameters{
  // Free normal beta to sum-to-zero normal beta under MVN
  matrix[C,M] beta;
  for(c in 1:C) {
    beta[c,] = to_row_vector(beta_raw[c]); //used with sum_to_zero_vector type
  }
  
  // Vectorized logarithm of variance-covariance matrix
  //matrix[A, (M-1)*(M-1)] log_Sigma_raw;
  matrix[A, (M*(M-1))%/%2] log_Sigma_raw;
  
  //for(a in 1:A) log_Sigma_raw[a] = to_row_vector(vectorized_matrix_log_spd(Sigma_raw[a]));
  for(a in 1:A) log_Sigma_raw[a] = to_row_vector(transform_cholesky_factor_cov(Sigma_raw[a])-vec_S_0);
  
  // Apply linear design to get unique sample-specific log VCoV
  //matrix[Ar, (M-1)*(M-1)] X_log_Sigma_raw = XA * log_Sigma_raw;
  matrix[Ar, (M*(M-1))%/%2] X_log_Sigma_raw = XA * log_Sigma_raw;
  
  // Take matrix exponential to get SPD VCoV back
  array[Ar] cholesky_factor_cov[M-1] L_X_Sigma_raw;
  //for(ar in 1:Ar) L_X_Sigma_raw[ar] = unvectorized_matrix_exp_spd(to_vector(X_log_Sigma_raw[ar]));
  for(ar in 1:Ar) L_X_Sigma_raw[ar] = inverse_transform_cholesky_factor_cov(to_vector(X_log_Sigma_raw[ar])+vec_S_0);
  
  // Non-centered parameterisation for intermediate u from Cholesky factors L_X_Sigma_raw
  matrix[N * !is_proportion, M] intermediate_u; // The actual residuals have dimension M
  if(!is_proportion){
      for(n in 1:N){
        intermediate_u[n,1:(M-1)] = (L_X_Sigma_raw[Xa_to_XA[n]]*to_vector(intermediate_u_raw[n]))';
        intermediate_u[n,M] = - sum(intermediate_u[n,1:(M-1)]);
        // variance-covariance of the last element is already determined by the previous elements
    }
  }
  
  // Non centered parameterisation SD of random effects
  array[M * (ncol_X_random_eff[1]> 0)] vector[how_many_factors_in_random_design[1]] random_effect_sigma;
  if(ncol_X_random_eff[1]> 0) for(m in 1:(M)) random_effect_sigma[m] = random_effect_sigma_mu[1] + random_effect_sigma_sigma[1] * random_effect_sigma_raw[m];
  if(ncol_X_random_eff[1]> 0) for(m in 1:(M)) random_effect_sigma[m] = exp(random_effect_sigma[m]/3.0);
  
  // Non centered parameterisation SD of random effects 2
  array[M * (ncol_X_random_eff[2]> 0)] vector[how_many_factors_in_random_design[2]] random_effect_sigma_2;
  if(ncol_X_random_eff[2]> 0) for(m in 1:(M)) random_effect_sigma_2[m] = random_effect_sigma_mu[2] + random_effect_sigma_sigma[2] * random_effect_sigma_raw_2[m];
  if(ncol_X_random_eff[2]> 0) for(m in 1:(M)) random_effect_sigma_2[m] = exp(random_effect_sigma_2[m]/3.0);
    
  matrix[ncol_X_random_eff[1] * (is_random_effect>0), M] random_effect;
  matrix[ncol_X_random_eff[2] * (is_random_effect>0), M] random_effect_2;
  
  // random intercept
  if(ncol_X_random_eff[1]> 0){
    
    // Convert sum_to_zero_vector array to vector array for function call
    array[ncol_X_random_eff[1]] vector[M] random_effect_raw_vec;
    for(i in 1:ncol_X_random_eff[1]) {
      random_effect_raw_vec[i] = to_vector(random_effect_raw[i]);
    }
    
    // Covariate setup
    random_effect =
    get_random_effect_matrix(
      M,
      n_groups[1],
      how_many_factors_in_random_design[1],
      is_random_effect,
      ncol_X_random_eff[1],
      group_factor_indexes_for_covariance,
      random_effect_raw_vec,
      random_effect_sigma,
      sigma_correlation_factor
      );
      
  }
  
  // random intercept
  if(ncol_X_random_eff[2]>0 ){

    // Convert sum_to_zero_vector array to vector array for function call
    array[ncol_X_random_eff[2]] vector[M] random_effect_raw_2_vec;
    for(i in 1:ncol_X_random_eff[2]) {
      random_effect_raw_2_vec[i] = to_vector(random_effect_raw_2[i]);
    }

    // Covariate setup
    random_effect_2 =
    get_random_effect_matrix(
      M,
      n_groups[2],
      how_many_factors_in_random_design[2],
      is_random_effect,
      ncol_X_random_eff[2],
      group_factor_indexes_for_covariance_2,
      random_effect_raw_2_vec,
      random_effect_sigma_2,
      sigma_correlation_factor_2
      );

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
      y_proportion,
      ysum,
      
      // Variance-Covariance
      L_X_Sigma_raw, // Only used when is_proportion
      intermediate_u, // Only used when !is_proportion
      Xa_to_XA, // Only used when is_proportion
      
      // Fixed effects
      X,                   
      beta, 
      M, 
      
      // Random effects
      ncol_X_random_eff,
      X_random_effect,   
      X_random_effect_2, 
      random_effect,
      random_effect_2
    );
  }
  
  // Priors
  if(exclude_priors == 0){
    // If intercept in design or I have complex variability design
    // This would include the models 
    // composition ~ 1 + ...; composition ~ 0 + ...; 
    // variability ~ 1
    if(intercept_in_design || A > 1){
      // Loop across the intercept columns in case of a intercept-less design (covariate are intercepts)
      for(a in 1:A_intercept_columns) Sigma_raw[a] ~ wishart_cholesky(prior_corr_eta + nu_lower, abundance_variability_regression(beta[a],prec_coeff,I_J,1));
      // Variability effect if the formula is more complex
      if(A>A_intercept_columns) for(a in (A_intercept_columns+1):A) Sigma_raw[a] ~ wishart_cholesky(prior_corr_eta + nu_lower, abundance_variability_regression(beta[a],prec_coeff,I_J,0));
    }
    else {
      Sigma_raw[1] ~ wishart_cholesky(prior_corr_eta + nu_lower, abundance_variability_regression(beta[1],prec_coeff,I_J,0));
    }
    
  }
  // When there is no mean-variability association, use the equal-correlation S_0 as S
  else{
    // Priors variability
    if(intercept_in_design || A > 1){
      for(a in 1:A_intercept_columns){
        Sigma_raw[a] ~ wishart_cholesky(prior_corr_eta + nu_lower, exp(prec_coeff[1])*S_0);
      }
      if(A>A_intercept_columns){
        for(a in (A_intercept_columns+1):A){
          Sigma_raw[a] ~ wishart_cholesky(prior_corr_eta + nu_lower,S_0);
        }
      }
    }
    // if ~ 0 + covariate
    else {
      Sigma_raw[1] ~ wishart_cholesky(prior_corr_eta + nu_lower, S_0);
    }
  }
  
  // Priors abundance - use correct scale for sum_to_zero_vector
  for(c in 1:B_intercept_columns) beta_raw[c] ~ normal ( prior_mean_intercept[1], prior_mean_intercept[2] * inv(sqrt(1 - inv(M))) );
  if(C>B_intercept_columns) for(c in (B_intercept_columns+1):C) beta_raw[c] ~ normal ( prior_mean_coefficients[1], prior_mean_coefficients[2] * inv(sqrt(1 - inv(M))) );

  // Hyper priors
  prec_coeff[1] ~ normal(prior_prec_intercept[1], prior_prec_intercept[2]);
  prec_coeff[2] ~ normal(prior_prec_slope[1],prior_prec_slope[2]);
  prec_sd ~ gamma(prior_prec_sd[1],prior_prec_sd[1]/prior_corr_eta); // 'eta' = prior_corr_eta/2+1

  // Priors for intermediate_u, only matters when using count data
  if(!is_proportion){
    to_vector(intermediate_u_raw) ~ std_normal();
  }

  // Random intercept
  if(is_random_effect>0){

    for(m in 1:M) random_effect_raw[,m] ~ normal(0, inv(sqrt(1 - inv(M)))); 
    for(m in 1:M) random_effect_sigma_raw[m] ~ std_normal();
    for(m in 1:M) sigma_correlation_factor[m] ~ lkj_corr_cholesky(2);   // LKJ prior for the correlation matrix

    random_effect_sigma_mu ~ std_normal();
    random_effect_sigma_sigma ~ std_normal();
    
    // If I have just one group
    zero_random_effect ~ std_normal();
  }
  if(ncol_X_random_eff[2]>0){
    for(m in 1:M) random_effect_raw_2[,m] ~ normal(0, inv(sqrt(1 - inv(M))));
    for(m in 1:M) random_effect_sigma_raw_2[m] ~ std_normal();
    for(m in 1:M) sigma_correlation_factor_2[m] ~ lkj_corr_cholesky(2);   // LKJ prior for the correlation matrix
    }
}

generated quantities {
  // Return complete singular VCoV as standard deviations and correlation matrix
  matrix[M, A*is_vb] sigma_as_sd;
  array[A*is_vb] matrix[M,M] Sigma;
  matrix[M, Ar*is_vb] Xsigma_as_sd;
  array[Ar*is_vb] matrix[M,M] XSigma;
  if(is_vb){
      for(a in 1:A){
          Sigma[a] = quad_form(multiply_lower_tri_self_transpose(Sigma_raw[a]),sum_to_zero_VCoV_transform_matrix(M-1));
          sigma_as_sd[,a] = sqrt(diagonal(Sigma[a]));
      }
      for(ar in 1:Ar){
          XSigma[ar] = quad_form(multiply_lower_tri_self_transpose(L_X_Sigma_raw[ar]),sum_to_zero_VCoV_transform_matrix(M-1));
          Xsigma_as_sd[,ar] = sqrt(diagonal(XSigma[ar]));
      }
  }
  
}

