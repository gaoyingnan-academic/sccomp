  // See https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factor-of-correlation-matrix-inverse-transform
  matrix inverse_transform_cholesky_factor_corr(vector y, int M){
    matrix[M,M] x = diag_matrix(rep_vector(1,M));
    if(M<=1) return x;
    if(M<=2){
      x[2,1] = y[1];
      return x;
    }
    for(i in 2:M){
      x[i,1] = tanh(y[1+((i-1)*(i-2))%/%2]);
      for(j in 2:(i-1)){
        x[i,j] = tanh(y[((i-1)*(i-2))%/%2+j])*sqrt(1-sum(square(x[i,1:(j-1)])));
        }
      x[i,i] = sqrt(1-sum(square(x[i,1:(i-1)])));
    }
    return x;
  }
  
  // See https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factor-of-correlation-matrix-transform
  vector transform_cholesky_factor_corr(matrix x, int M){
    vector[(M*(M-1))%/%2] y;
    if(M<=1) return y; // Returns a vector of length zero
    if(M<=2){
      y[1] = x[2,1];
      return y;
    }
    for(i in 2:M){
      y[((i-1)*(i-2))%/%2+1] = x[i,1];
      for(j in 2:(i-1)){
        y[((i-1)*(i-2))%/%2+j] = x[i,j]/sqrt(1-sum(square(x[i,1:j-1])));
        }
    }
    return 0.5*(log(1+y)-log(1-y));// tanh^-1 or arctanh
  }
  
  // If I transform Cholesky factor of a VCoV matrix to a vector
  vector transform_cholesky_factor_cov(matrix L_Sigma){
    int M = cols(L_Sigma);
    vector[M] sigma = sqrt(diagonal(multiply_lower_tri_self_transpose(L_Sigma)));
    matrix[M,M] L_R = diag_pre_multiply((inv(sigma)),L_Sigma);
    vector[(M*(M+1))%/%2] vectorized_L_Sigma;
    vectorized_L_Sigma[1:M] = log(sigma);
    vectorized_L_Sigma[(1+M):((M*(M+1))%/%2)] = transform_cholesky_factor_corr(L_R, M);
    return vectorized_L_Sigma;
  }
  
  // Inverse transformation to VCoV Cholesky factor
  matrix inverse_transform_cholesky_factor_cov(vector vec_L_Sigma){
    int M = to_int(sqrt(2*num_elements(vec_L_Sigma)));
    vector[M] sigma = exp(vec_L_Sigma[1:M]);
    matrix[M,M] L_R = inverse_transform_cholesky_factor_corr(vec_L_Sigma[(M+1):num_elements(vec_L_Sigma)],M);
    matrix[M,M] L_Sigma = diag_pre_multiply(sigma,L_R);
    return L_Sigma;
  }
  