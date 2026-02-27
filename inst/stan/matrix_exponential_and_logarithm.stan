  // Specialized matrix logarithm for SPD matrices
  vector vectorized_matrix_log_spd(matrix B){
    int M = cols(B);
    vector[M] lambda; //eigenvalues
    matrix[M,M] Q; // matrix of column-eigenvectors
    matrix[M,M] log_B;
    (Q,lambda) = eigendecompose_sym(B);
    for(m in 1:M) lambda[m] = fmax(lambda[m], 1e-12); //enforce strict positive eigenvalues
    log_B = Q*diag_matrix(log(lambda))*Q';
    return to_vector(log_B); // return vectorized matrix for internal use
    // technically only the lower/upper triangle part with the diagonal is required
  }
  
  // Specialized matrix exponential for SPD matrices
  matrix unvectorized_matrix_exp_spd(vector b){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    vector[M] lambda; //eigenvalues
    matrix[M,M] Q; // matrix of column-eigenvectors
    matrix[M,M] exp_B;
    (Q,lambda) = eigendecompose_sym(B);
    for(m in 1:M) lambda[m] = fmax(lambda[m], 1e-12); //enforce strict positive eigenvalues
    exp_B = Q*diag_matrix(exp(lambda))*Q';
    return cholesky_decompose(exp_B); // return Cholesky factor for internal use 
  }
