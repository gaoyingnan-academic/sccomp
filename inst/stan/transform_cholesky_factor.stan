  // See https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factor-of-correlation-matrix-inverse-transform
  matrix inverse_transform_cholesky_factor_corr(vector y, int M){
    matrix[M,M] x = diag_matrix(rep_vector(1,M));
    if(M<=1) return x;
    for(i in 2:M){
      x[i,1] = tanh(y[i*(i-1)%/%2]);
      for(j in 2:i){
        x[i,j] = tanh(y[i*(i-1)%/%2+j-1])*sqrt(1-sum(square(x[i,1:(j-1)])));
        }
    }
    return x;
  }
  
  // See https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factor-of-correlation-matrix-transform
  vector transform_cholesky_factor_corr(matrix x, int M){
    vector[M*(M-1)%/%2] y;
    if(M<=1) return y; // Returns a vector of length zero
    for(i in reverse(linspaced_int_array(M,2,M-1))){
      y[i*(i-1)%/%2] = x[i,1];
      for(j in reverse(linspaced_int_array(i,2,i-1))){
        y[i*(i-1)%/%2+j-1] = x[i,j]/sqrt(1-sum(square(x[i,1:j-1])));
        }
    }
    return 0.5*(log(1+y)-log(1-y));// tanh^-1 or arctanh
  }