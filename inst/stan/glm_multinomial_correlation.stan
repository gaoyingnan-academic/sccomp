functions{
 
  #include common_functions.stan
  #include transform_cholesky_factor.stan
  array[] int rep_each(array[] int x, int K) {
    int N = size(x);
    array[N * K] int y;
    int pos = 1;
    for (n in 1:N) {
      for (k in 1:K) {
        y[pos] = x[n];
        pos += 1;
      }
    }
    return y;
  }
  
  real abundance_variability_regression(row_vector variability, row_vector abundance, array[] real prec_coeff, real prec_sd, int bimodal_mean_variability_association, real mix_p){
    
    real lp = 0;
    // If mean-variability association is bimodal such as for single-cell RNA use mixed model
    if(bimodal_mean_variability_association == 1){
      for(m in 1:cols(variability))
      lp += log_mix(mix_p,
      normal_lpdf(variability[m] | abundance[m] * prec_coeff[2] + prec_coeff[1], prec_sd ),
      normal_lpdf(variability[m] | abundance[m] * prec_coeff[2] + 1, prec_sd)  // -0.73074903 is what we observe in single-cell dataset Therefore it is safe to fix it for this mixture model as it just want to capture few possible outlier in the association
      );
      
      // If no bimodal
    } else {
      lp =  normal_lpdf(variability | abundance * prec_coeff[2] + prec_coeff[1], prec_sd );
    }
    
    return(lp);
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
    
    // Precision
    matrix precision,                   // Sliced
    
    // Correlation
    matrix transformed_Xa_L_Omega, // Sliced
    matrix intermediate_u, // Sliced
    
    // Fixed effects
    matrix X,                   // Sliced
    matrix beta, 
    int M, 
    
    // Random effects
    array[] int ncol_X_random_eff,
    matrix X_random_effect,   // Sliced
    matrix X_random_effect_2,  // Sliced
    matrix random_effect,
    matrix random_effect_2,
    
    ){
      
      int N = end-start+1; // Number of observations subsetted to this chunk
      
      // mu
      matrix[M, N] mu = (X[idx_y,] * beta)';
      
      if(ncol_X_random_eff[1]> 0)
      mu = mu + (X_random_effect[idx_y,] * random_effect)';
      
      if(ncol_X_random_eff[2]>0 )
      mu = mu + (X_random_effect_2[idx_y,] * random_effect_2)';
      
      if(!is_proportion){
        mu = mu + intermediate_u[idx_y,]';
        for(n in 1:N){
          mu[,n] = softmax(mu[,n]);
        }
      }

      // Precision
      matrix[M, N] precision = exp((Xa[idx_y,] * alpha)');
      
      // target log-probability as multivariate functions are not vectorized with regard to Cholesky factors
      real target_lp = 0;

      // If input is proportions
      if(is_proportion){
        for(n in 1:N){
          target_lp += multi_normal_cholesky_lupdf(
            to_vector(y_proportion[idx_y[n],]) |
            mu[,idx_y[n]],
            inverse_transform_cholesky_factor_corr(
              to_vector(transformed_Xa_L_Omega[idx_y[n]]),M)
          );
        }
        return target_lp;
      }
        // If input is counts
        else{
          for(n in 1:N){
            target_lp += multinomial_lupmf(
              to_array_1d(y[idx_y[n],]) |
              mu[,idx_y[n]]
            );
          }
          return target_lp;
        }
    }
    
}

data{
  int<lower=0, upper=1> is_proportion;
  int<lower=1> N; // Equivalent to S in publication
  int<lower=1> M; // Equivalent to G in publication
  int<lower=1> C;
  int<lower=1> A; // How many column in variability design\
  int<lower=1> A_intercept_columns; // How many intercept column in varibility design
  int<lower=1> B_intercept_columns; // How many intercept column in varibility design
  int<lower=1> Ar; // Rows of unique variability design
  int<lower=1> R; // How many column in correlation design
  array[N] int exposure;
  array[N * !is_proportion,M] int<lower=0> y;
  array[N * is_proportion,M] real<lower=0, upper=1> y_proportion;
  matrix[N, C] X;
  matrix[Ar, A] XA; // The unique variability design
  matrix[N, A] Xa; // The variability design
  matrix[N, R] Xr; // The correlation design
  
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
  
  // Prior info
  array[2] real prior_prec_intercept;
  array[2] real prior_prec_slope;
  array[2] real prior_prec_sd;
  array[2] real prior_mean_intercept;
  array[2] real prior_mean_coefficients;
  
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
  
}

transformed data{
  // EXCEPTION MADE FOR WINDOWS GENERATE QUANTITIES IF RANDOM EFFECT DO NOT EXIST
  int ncol_X_random_eff_WINDOWS_BUG_FIX = max(ncol_X_random_eff[1], 1);
  int ncol_X_random_eff_WINDOWS_BUG_FIX_2 = max(ncol_X_random_eff[2], 1);
  
  // For parallelisation
  array[N] int array_N;
  for(n in 1:N) array_N[n] = n;
  
  // For proportional data
  array[N * is_proportion,M] real y_proportion_clr_transformed;
  if(is_proportion){
    y_proportion_clr_transformed = log(y_proportion);
    for(n in 1:N){
      y_proportion_clr_transformed[n] = y_proportion_clr_transformed[n] - mean(y_proportion_clr_transformed[n]);
    }
  }
  
}

parameters{
  // Use the new sum_to_zero_vector type instead of QR decomposition
  array[C] sum_to_zero_vector[M] beta_raw; // Each row is a sum_to_zero_vector of length M
  matrix[A, M] alpha; // Variability
  
  // New parameters for correlation
  array[A] cholesky_factor_corr[M] L_Omega; // Cholesky factor for correlation matrices
  matrix[N * !is_proportion, M] intermediate_u_raw; // independent components of correlated residuals to bridge proportions and read counts
  
  // To exclude
  array[2] real prec_coeff;
  real<lower=0> prec_sd;
  real<lower=0, upper=1> mix_p;
  
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
  
  // Initialisation
  matrix[C,M] beta;
  matrix[M, N] precision = exp((Xa * alpha)');
  
  // Transform Cholesky factors to vectors so they can multiply with the design matrix
  matrix[A, (M*(M-1))%/%2] transformed_L_Omega; // For unconstrained operations on Cholesky factors
  for(aa in 1:A){
    transformed_L_Omega[aa] = 
      to_row_vector(transform_cholesky_factor_corr(L_Omega[aa],M));
  }
  matrix[N, (M*(M-1))%/%2] transformed_Xa_L_Omega = Xa * transformed_L_Omega;
  
  // Inverse-transform the vectors back to Cholesky factors
  array[N] matrix[M,M] Xa_L_Omega; // inverse-transformed from unconstrained values
  for(n in 1:N){
    Xa_L_Omega[n] = 
      inverse_transform_cholesky_factor_corr(
        to_vector(transformed_Xa_L_Omega[n]),M);
  }
  matrix[N * !is_proportion, M] intermediate_u;

  // Convert sum_to_zero_vector to regular matrix
  for(c in 1:C) {
    beta[c,] = to_row_vector(beta_raw[c]);
  }
  
  // Non-centered parameterisation for intermediate u
  if(!is_proportion){
      for(n in 1:N){
        intermediate_u[n] = (diag_pre_multiply(precision[,n],Xa_L_Omega[n])*to_vector(intermediate_u_raw[n]))';
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
      exposure,  
      
      // Precision
      Xa,                   
      alpha,
      transformed_Xa_L_Omega, // Only used when is_proportion
      intermediate_u, // Only used when !is_proportion
      
      // Fixed effects
      X,                   
      beta, 
      M, 
      
      // Random effects
      ncol_X_random_eff,
      X_random_effect,   
      X_random_effect_2, 
      random_effect,
      random_effect_2,
      
      //truncation
      truncation_not_idx_minimal
      
      );
      
      // print("2---", reduce_sum(
        //   partial_sum_lupmf,
        //   y_array[truncation_not_idx],
        //   grainsize,
        //   exposure_array[truncation_not_idx],
        //   mu_array[truncation_not_idx],
        //   precision_array[truncation_not_idx]
        //   ));
        
        
  }
  
  // Priors
  if(exclude_priors == 0){
    
    // If interceopt in design or I have complex variability design
    // This would include the models 
    // composition ~ 1 + ...; composition ~ 0 + ...; 
    // variability ~ 1
    if(A == 1){
      target += abundance_variability_regression(
        alpha[1],
        beta[1], // average_by_col(beta[1:B_intercept_columns,]),
        prec_coeff,
        prec_sd,
        bimodal_mean_variability_association,
        mix_p
        );
    }
    else {
      // Loop across the intercept columns in case of a intercept-less design (covariate are intercepts)
      for(a in 1:A_intercept_columns)
      target += abundance_variability_regression(
        alpha[a],
        beta[a],
        prec_coeff,
        prec_sd,
        bimodal_mean_variability_association,
        mix_p
        );
        
        // Variability effect if the formula is more complex
        if(A>A_intercept_columns) for(a in (A_intercept_columns+1):A) alpha[a] ~ normal(beta[a] * prec_coeff[2], 2 );
    }
    
  }
  
  // If I don't have priors for overdispersion
  else{
    // Priors variability
    if(intercept_in_design || A > 1){
      for(a in 1:A_intercept_columns) alpha[a]  ~ normal( prec_coeff[1], prec_sd );
      if(A>A_intercept_columns) for(a in (A_intercept_columns+1):A) to_vector(alpha[a]) ~ normal ( 0, 2 );
    }
    // if ~ 0 + covariuate
    else {
      alpha[1]  ~ normal( prec_coeff[1], prec_sd );
    }
  }
  
  // // Priors abundance - use correct scale for sum_to_zero_vector
  for(c in 1:B_intercept_columns) beta_raw[c] ~ normal ( prior_mean_intercept[1], prior_mean_intercept[2] * inv(sqrt(1 - inv(M))) );
  if(C>B_intercept_columns) for(c in (B_intercept_columns+1):C) beta_raw[c] ~ normal ( prior_mean_coefficients[1], prior_mean_coefficients[2] * inv(sqrt(1 - inv(M))) );
  
  // Hyper priors
  mix_p ~ beta(1,5);
  prec_coeff[1] ~ normal(prior_prec_intercept[1], prior_prec_intercept[2]);
  prec_coeff[2] ~ normal(prior_prec_slope[1],prior_prec_slope[2]);
  prec_sd ~ gamma(prior_prec_sd[1],prior_prec_sd[2]);
  prec_coeff ~ std_normal();
  // Note: sum_to_zero_vector has built-in priors, no need for explicit std_normal()
  
  // (Hyper-)priors for the correlation matrices
  for(aa in 1:A){
      L_Omega[aa] ~ lkj_corr_cholesky(2);
  }
  // Priors for intermediate_u, only matters when using count data
  if(!is_proportion){
      for(n in 1:N){
        intermediate_u_raw[n] ~ normal(0.0, 1.0);
    }
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
  matrix[A, M] alpha_normalised = alpha;
  
  // // Rondom effect
  // matrix[ncol_X_random_eff_WINDOWS_BUG_FIX, M] beta_random_effect;
  // matrix[ncol_X_random_eff_WINDOWS_BUG_FIX_2, M] beta_random_effect_2;
  
  // LOO
  vector[TNS] log_lik = rep_vector(0, TNS);
  
  // These instructions regress out the effect of mean proportion to the overdispersion
  // This adjustment provide A overdispersion value that can be tested for a hypotheses for example differences between two conditions
  if(intercept_in_design){
    if(A > 1) for(a in 2:A) alpha_normalised[a] = alpha[a] - (beta[a] * prec_coeff[2] );
  }
  else{
    for(a in 1:A) alpha_normalised[a] = alpha[a] - (beta[a] * prec_coeff[2] );
  }
  
  // LOO
  if(enable_loo==1){

    matrix[M, N] mu;
    vector[N*M] mu_array;
    vector[N*M] precision_array;

    mu = (X * beta)';

    // random intercept
    if(ncol_X_random_eff[1]> 0)
    mu = mu + (X_random_effect * random_effect)';
    if(ncol_X_random_eff[2]>0 )
    mu = mu + (X_random_effect_2 * random_effect_2)';


    // Calculate proportions
    for(n in 1:N)  mu[,n] = softmax(mu[,n]);

    // Convert the matrix m to a column vector in column-major order.
    mu_array = to_vector(mu);
    precision_array = to_vector(exp(precision));

    if(is_proportion)
          for (n in 1:TNS) {
      log_lik[n] = beta_lpdf(
        to_array_1d(y_proportion)[truncation_not_idx[n]] |
        (mu_array[truncation_not_idx[n]] .* precision_array[truncation_not_idx[n]]),
        ((1.0 - mu_array[truncation_not_idx[n]]) .* precision_array[truncation_not_idx[n]])
        ) ;
      }
    else
      for (n in 1:TNS) {
       log_lik[n] = beta_binomial_lpmf(
        to_array_1d(y)[truncation_not_idx[n]] |
        rep_each(exposure, M)[truncation_not_idx[n]],
        (mu_array[truncation_not_idx[n]] .* precision_array[truncation_not_idx[n]]),
        ((1.0 - mu_array[truncation_not_idx[n]]) .* precision_array[truncation_not_idx[n]])
        ) ;
    }

  }
}

