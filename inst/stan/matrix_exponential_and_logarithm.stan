  // Specialized matrix logarithm for SPD matrices
  vector vectorized_matrix_log_spd(matrix B){
    int M = cols(B);
    vector[M] lambda; //eigenvalues
    matrix[M,M] Q; // matrix of column-eigenvectors
    matrix[M,M] log_B;
    (Q,lambda) = eigendecompose_sym(B);
    log_B = Q*diag_matrix(log(lambda))*Q';
    return to_vector(log_B); // return vectorized matrix for internal use
    // technically only the lower/upper triangle part with the diagonal is required
  }
  
  // Specialized matrix exponential for SPD matrices
  matrix unvectorized_matrix_exp_spd(vector b){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    B = (B+B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    vector[M] lambda; //eigenvalues
    matrix[M,M] Q; // matrix of column-eigenvectors
    matrix[M,M] exp_B;
    (Q,lambda) = eigendecompose_sym(B);
    exp_B = Q*diag_matrix(exp(lambda))*Q';
    exp_B = (exp_B+exp_B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    return cholesky_decompose(exp_B); // return Cholesky factor for internal use
  }

  // Matrix exponential using degree-13 Pade approximant
  matrix unvectorized_matrix_exp_pade(vector b){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    B = (B+B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    vector[M] lambda; //eigenvalues
    matrix[M,M] Q; // matrix of column-eigenvectors
    matrix[M,M] exp_B;
    (Q,lambda) = eigendecompose_sym(B);
    exp_B = Q*diag_matrix(exp(lambda))*Q';
    exp_B = (exp_B+exp_B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    return cholesky_decompose(exp_B); // return Cholesky factor for internal use
  }
  
  // Matrix exponential using scaling and truncated Taylor
  matrix unvectorized_matrix_exp_taylor(vector b, int K, int s){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    B = (B+B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    matrix[M,M] X = B/pow(2.0,s); // scaled matrix whose exponent will be approximated by Taylor
    matrix[M,M] EX = diag_matrix(rep_vector(1.0,M)); // exponent approximant starting from identity
    matrix[M,M] EX_increment = EX;
    for(k in 1:K){
      EX_increment = EX_increment * X / k; // Taylor's expansion
      EX += EX_increment;
    }
    // scale EX back to B
    for(k in 1:s){
      EX = EX * EX;
    }
    EX = (EX + EX') / 2; // To avoid 'A is not symmetric' problem due to floating error.
    return cholesky_decompose(EX); // return Cholesky factor for internal use
  }

  // assemble symmetric matrix from sampled diagonal and off-diagonal vector
  matrix sym_matrix_from_vectors(vector diag_b, vector off_diag_b){
    int M = num_elements(diag_b); // length of off_diag_b not checked for internal use
    matrix[M,M] B = diag_matrix(diag_b);
    int i;
    int j = 0;
    // construct symmetric matrix by subdiagonal
    for(m in 2:M){
      i = j+1;
      j = i+M-m;
      B[m:M,1:(M-m+1)]+=diag_matrix(off_diag_b[i:j]);
      B[1:(M-m+1),m:M]+=diag_matrix(off_diag_b[i:j]);
    }
    return B;
  }
