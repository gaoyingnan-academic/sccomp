functions{
 
  #include common_functions.stan
  #include transform_cholesky_factor.stan
  #include singular_VCoV_matrix.stan
  
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
    array[,] int y, //Sliced, M dimensions
    array[,] real y_proportion, // ILR, M-1 dimensions
    array[] int ysum, //Sliced,
    
    // Variance-Covariance
    array[] matrix Lhat, // ILR, M-1 dimensions
    matrix intermediate_u, // Sliced, ILR
    array[] int Xa_to_XA, // Sliced
    
    // Fixed effects
    matrix X,                   // Sliced
    matrix beta, // ILR-transformed, M-1 dimensions (beta_raw) 
    int M, 
    
    // Random effects
    array[] int ncol_X_random_eff,
    matrix X_random_effect,   // Sliced
    matrix X_random_effect_2,  // Sliced
    matrix random_effect,
    matrix random_effect_2,
    
    // Helmert matrix
    matrix Helmert
    ){
      
      int N = end-start+1; // Number of observations subsetted to this chunk
      
      // mu
      matrix[M-1, N] mu = (X[idx_y,] * beta)';
      
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
            to_vector(y_proportion[idx_y[n]]) |
            mu[,idx_y[n]],
            to_matrix(Lhat[Xa_to_XA[idx_y[n]]])
          );
        }
      }
      // If input is counts
      else{
        matrix[M,N] MU = Helmert*(mu + intermediate_u[idx_y,]'); // Back to CLR
        for(n in 1:N){
          MU[,n] = softmax(MU[,n]); // Then to simplex
        }
        // GPU-compatible pseudo-vectorization
        target_lp += poisson_lupmf(
          to_array_1d(y[idx_y,]) |
          to_vector(diag_post_multiply(MU,to_vector(ysum[idx_y])))
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
  
  // Prior info
  array[2] real prior_prec_intercept;
  array[2] real prior_prec_slope;
  array[2] real prior_prec_sd;
  array[2] real prior_mean_intercept;
  array[2] real prior_mean_coefficients;
  real<lower=0> prior_corr_eta; 
  
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
  matrix[M,M-1] Helmert;
  
}

transformed data{
  // centered and isometric log-ratio transformed data (only relevant for proportional input)
  matrix<lower=0, upper=1>[N * is_proportion,M] clr_y_proportion;
  matrix<lower=0, upper=1>[N * is_proportion,M-1] ilr_y_proportion;
  
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
  
  // EXCEPTION MADE FOR WINDOWS GENERATE QUANTITIES IF RANDOM EFFECT DO NOT EXIST
  int ncol_X_random_eff_WINDOWS_BUG_FIX = max(ncol_X_random_eff[1], 1);
  int ncol_X_random_eff_WINDOWS_BUG_FIX_2 = max(ncol_X_random_eff[2], 1);
  
  // For CPU parallelisation
  array[N] int array_N;
  for(n in 1:N) array_N[n] = n;

}

parameters{
  // // ILR means are not constrained
  matrix[M-1,C] beta_raw; // Only M-1 degree of freedom

  // ILR log-standard deviations, not constrained
  matrix[A, M-1] alpha_raw;
  
  // ILR Cholesky factors for correlation matrices
  array[A] cholesky_factor_corr[M-1] L; // Cholesky factor for correlation matrices
  
  // raw ILR latent residuals
  matrix[N * use_data, (M-1) * !is_proportion] intermediate_u_raw;
  
  // To exclude
  array[2] real prec_coeff;
  real<lower=0> prec_sd;
  real<lower=0, upper=1> mix_p;
  
  // Random intercept in ILR space
  array[ncol_X_random_eff[1] * (is_random_effect>0)] vector[M-1] random_effect_raw;
  array[ncol_X_random_eff[2] * (ncol_X_random_eff[2]>0)] vector[M-1] random_effect_raw_2;
  
  // sd of random intercept
  array[2 * (is_random_effect>0)] real random_effect_sigma_mu;
  array[2 * (is_random_effect>0)] real random_effect_sigma_sigma;
  
  // Covariance
  array[(M-1) * (is_random_effect>0)] vector[how_many_factors_in_random_design[1]]  random_effect_sigma_raw;
  array[(M-1) * (is_random_effect>0)] cholesky_factor_corr[how_many_factors_in_random_design[1] * (is_random_effect>0)] sigma_correlation_factor;
  
  // Covariance
  array[(M-1) * (is_random_effect>0)] vector[how_many_factors_in_random_design[2]]  random_effect_sigma_raw_2;
  array[(M-1) * (is_random_effect>0)] cholesky_factor_corr[how_many_factors_in_random_design[2] * (is_random_effect>0)] sigma_correlation_factor_2;
  
  // If I have just one group
  array[is_random_effect>0] real zero_random_effect;
  
}

transformed parameters{
  // ILR variance-covariance
  matrix[M-1, Ar] sigma_raw = exp((XA*alpha_raw)');
  
  // Transform ILR Cholesky factors to vectors so they can multiply with the design matrix
  matrix[A, ((M-2)*(M-1))%/%2] transformed_L; // For unconstrained linear operations on Cholesky factors
  for(a in 1:A){
    transformed_L[a] = 
      to_row_vector(transform_cholesky_factor_corr(L[a],M-1));
  }
  matrix[Ar, ((M-2)*(M-1))%/%2] transformed_Lhat = XA * transformed_L; //Design-specific Cholesky factors
  
  // Inverse-transform the vectors back to Cholesky factors (ILR)
  array[Ar] matrix[M-1,M-1] Lhat; // inverse-transformed from unconstrained values
  for(ar in 1:Ar){
    Lhat[ar] = 
      inverse_transform_cholesky_factor_corr(to_vector(transformed_Lhat[ar]),M-1);
    Lhat[ar] = diag_pre_multiply(sigma_raw[,ar],Lhat[ar]); // VCoV Cholesky
  }
  
  // Non-centered parameterisation for ILR intermediate u
  matrix[N * use_data , (M-1) * !is_proportion] intermediate_u; // ILR
  if(use_data&&(!is_proportion)){
      for(n in 1:N){
        intermediate_u[n] = to_row_vector(Lhat[Xa_to_XA[n]]*to_vector(intermediate_u_raw[n]));
    }
  }
  
  // Non centered parameterisation SD of random effects
  array[(M-1) * (ncol_X_random_eff[1]> 0)] vector[how_many_factors_in_random_design[1]] random_effect_sigma;
  if(ncol_X_random_eff[1]> 0) for(m in 1:(M-1)) random_effect_sigma[m] = random_effect_sigma_mu[1] + random_effect_sigma_sigma[1] * random_effect_sigma_raw[m];
  if(ncol_X_random_eff[1]> 0) for(m in 1:(M-1)) random_effect_sigma[m] = exp(random_effect_sigma[m]/3.0);
  
  // Non centered parameterisation SD of random effects 2
  array[(M-1) * (ncol_X_random_eff[2]> 0)] vector[how_many_factors_in_random_design[2]] random_effect_sigma_2;
  if(ncol_X_random_eff[2]> 0) for(m in 1:(M-1)) random_effect_sigma_2[m] = random_effect_sigma_mu[2] + random_effect_sigma_sigma[2] * random_effect_sigma_raw_2[m];
  if(ncol_X_random_eff[2]> 0) for(m in 1:(M-1)) random_effect_sigma_2[m] = exp(random_effect_sigma_2[m]/3.0);
    
  
  matrix[ncol_X_random_eff[1] * (is_random_effect>0), M-1] random_effect;
  matrix[ncol_X_random_eff[2] * (is_random_effect>0), M-1] random_effect_2;
  
  // random intercept
  if(ncol_X_random_eff[1]> 0){
    
    // Conversion to vector array no longer required, but kept for convenience
    array[ncol_X_random_eff[1]] vector[M-1] random_effect_raw_vec;
    for(i in 1:ncol_X_random_eff[1]) {
      random_effect_raw_vec[i] = to_vector(random_effect_raw[i]);
    }
    
    // Covariate setup
    random_effect =
    get_random_effect_matrix(
      M-1,
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

    // Conversion to vector array no longer required, but kept for convenience
    array[ncol_X_random_eff[2]] vector[M-1] random_effect_raw_2_vec;
    for(i in 1:ncol_X_random_eff[2]) {
      random_effect_raw_2_vec[i] = to_vector(random_effect_raw_2[i]);
    }

    // Covariate setup
    random_effect_2 =
    get_random_effect_matrix(
      M-1,
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
      to_array_2d(ilr_y_proportion),
      ysum,
      
      // Variance-Covariance
      Lhat, // Only used when is_proportion
      intermediate_u, // Only used when !is_proportion
      Xa_to_XA, // Only used when is_proportion
      
      // Fixed effects
      X,                   
      beta_raw', 
      M, 
      
      // Random effects
      ncol_X_random_eff,
      X_random_effect,   
      X_random_effect_2, 
      random_effect,
      random_effect_2,
      
      // Helmert matrix
      Helmert
      );
  }
  
  // Priors
  if(exclude_priors == 0){
    
    // If interceopt in design or I have complex variability design
    // This would include the models 
    // composition ~ 1 + ...; composition ~ 0 + ...; 
    // variability ~ 1
    if(A == 1){
      target += abundance_variability_regression(
        alpha_raw[1],
        to_row_vector(beta_raw[,1]),
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
        alpha_raw[a],
        to_row_vector(beta_raw[,a]),
        prec_coeff,
        prec_sd,
        bimodal_mean_variability_association,
        mix_p
        );
        
        // Variability effect if the formula is more complex
        if(A>A_intercept_columns) for(a in (A_intercept_columns+1):A) alpha_raw[a] ~ normal(beta_raw[,a] * prec_coeff[2], 2 );
    }
    
  }
  
  // If I don't have priors for overdispersion
  else{
    // Priors variability
    if(intercept_in_design || A > 1){
      for(a in 1:A_intercept_columns) alpha_raw[a]  ~ normal( prec_coeff[1], prec_sd );
      if(A>A_intercept_columns) for(a in (A_intercept_columns+1):A) to_vector(alpha_raw[a]) ~ normal ( 0, 2 );
    }
    // if ~ 0 + covariate
    else {
      alpha_raw[1]  ~ normal( prec_coeff[1], prec_sd );
    }
  }
  
  // Priors abundance - use correct scale for sum_to_zero_vector
  for(c in 1:B_intercept_columns) beta_raw[c] ~ normal ( prior_mean_intercept[1], prior_mean_intercept[2]);
  if(C>B_intercept_columns) for(c in (B_intercept_columns+1):C) beta_raw[c] ~ normal ( prior_mean_coefficients[1], prior_mean_coefficients[2]);
  // // Priors abundance - use mvn in transformed parameters to avoid loop calls of prior
  //to_vector(beta_raw) ~ normal(0,1);
  
  // Hyper priors
  mix_p ~ beta(1,5);
  prec_coeff[1] ~ normal(prior_prec_intercept[1], prior_prec_intercept[2]);
  prec_coeff[2] ~ normal(prior_prec_slope[1],prior_prec_slope[2]);
  prec_sd ~ gamma(prior_prec_sd[1],prior_prec_sd[2]);
  prec_coeff ~ std_normal();
  // Note: sum_to_zero_vector has built-in priors, no need for explicit std_normal()
  
  // (Hyper-)priors for the correlation matrices
  for(a in 1:A){
      L[a] ~ lkj_corr_cholesky(prior_corr_eta);
  }
  // Priors for intermediate_u, only matters when using count data
  if(use_data&&(!is_proportion)){
    to_vector(intermediate_u_raw) ~ std_normal();
  }

  // Random intercept
  if(is_random_effect>0){

    for(m in 1:(M-1)) random_effect_raw[,m] ~ std_normal(); 
    for(m in 1:(M-1)) random_effect_sigma_raw[m] ~ std_normal();
    for(m in 1:(M-1)) sigma_correlation_factor[m] ~ lkj_corr_cholesky(2);   // LKJ prior for the correlation matrix

    random_effect_sigma_mu ~ std_normal();
    random_effect_sigma_sigma ~ std_normal();
    
    // If I have just one group
    zero_random_effect ~ std_normal();
  }
  if(ncol_X_random_eff[2]>0){
    for(m in 1:(M-1)) random_effect_raw_2[,m] ~ std_normal(); 
    for(m in 1:(M-1)) random_effect_sigma_raw_2[m] ~ std_normal();
    for(m in 1:(M-1)) sigma_correlation_factor_2[m] ~ lkj_corr_cholesky(2);   // LKJ prior for the correlation matrix
    }
    
}

generated quantities {
  // ILR means to CLR means for interpretability
  matrix[C,M] beta = (Helmert*beta_raw)';

  // Return complete singular VCoV as standard deviations and correlation matrix
  matrix[M*is_vb, A*is_vb] full_alpha;
  array[A*is_vb] matrix[M,M] full_L;
  matrix[M*is_vb, Ar*is_vb] full_sigma;
  array[Ar*is_vb] matrix[M,M] full_Omega;
  if(is_vb){
      for(a in 1:A){
          (full_alpha[,a],full_L[a]) = get_std_and_corr_of_VCoV(
            Helmert*multiply_lower_tri_self_transpose(
              diag_pre_multiply(exp(to_vector(alpha_raw[a])),L[a])
              )*Helmert'
            );
      }
      for(ar in 1:Ar){
          (full_sigma[,ar],full_Omega[ar]) = get_std_and_corr_of_VCoV(
            Helmert*multiply_lower_tri_self_transpose(Lhat[ar])*Helmert'
            );
      }
  }
  
  //matrix[A, M] alpha_normalised = alpha;
  
  // // Rondom effect
  // matrix[ncol_X_random_eff_WINDOWS_BUG_FIX, M] beta_random_effect;
  // matrix[ncol_X_random_eff_WINDOWS_BUG_FIX_2, M] beta_random_effect_2;
  
  /**
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
  **/
  
  /**
  // LOO
  if(enable_loo==1){

    matrix[M, N] mu;
    vector[N*M] mu_array;
    vector[N*M] sigma_array;

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
    sigma_array = to_vector(exp(sigma));

    if(is_proportion)
          for (n in 1:TNS) {
      log_lik[n] = beta_lpdf(
        to_array_1d(y_proportion)[truncation_not_idx[n]] |
        (mu_array[truncation_not_idx[n]] .* sigma_array[truncation_not_idx[n]]),
        ((1.0 - mu_array[truncation_not_idx[n]]) .* sigma_array[truncation_not_idx[n]])
        ) ;
      }
    else
      for (n in 1:TNS) {
       log_lik[n] = beta_binomial_lpmf(
        to_array_1d(y)[truncation_not_idx[n]] |
        rep_each(exposure, M)[truncation_not_idx[n]],
        (mu_array[truncation_not_idx[n]] .* sigma_array[truncation_not_idx[n]]),
        ((1.0 - mu_array[truncation_not_idx[n]]) .* sigma_array[truncation_not_idx[n]])
        ) ;
    }

  }
  **/
}

