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
