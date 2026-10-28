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
  matrix unvectorized_matrix_exp_pade(vector b, int s){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    matrix[M,M] I = diag_matrix(rep_vector(1.0,M));
    
    B = (B+B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    real mu_B = sum(diagonal(B))/M;
    B = B - mu_B*I; // Reduce norm for better precision
    
    matrix[M,M] X = B/pow(2.0,s); // scaled matrix whose exponent will be approximated by Pade approximant
    matrix[M,M] X2 = X*X;
    matrix[M,M] X4 = X2*X2;
    matrix[M,M] X6 = X4*X2;
    
    matrix[M,M] U;
    matrix[M,M] V;
    matrix[M,M] R;
    
    // Degree-13 Pade numerator: odd terms
    U = X * (
      X6 * (
        1.0 * X6
        + 16380.0 * X4
        + 40840800.0 * X2
      )
      + 33522128640.0 * X6
      + 10559470521600.0 * X4
      + 1187353796428800.0 * X2
      + 32382376266240000.0 * I
    );

    // Degree-13 Pade denominator: even terms
    V = X6 * (
        182.0 * X6
        + 960960.0 * X4
        + 1323241920.0 * X2
      )
      + 670442572800.0 * X6
      + 129060195264000.0 * X4
      + 7771770303897600.0 * X2
      + 64764752532480000.0 * I;
    
    // Solve (V - U) * R = V + U
    R = mdivide_left(V - U, V + U);
    
    // scale R back to B
    for(k in 1:s){
      R = R * R;
    }
    R = (R + R') / 2; // To avoid 'A is not symmetric' problem due to floating error.
    return exp(mu_B/2)*cholesky_decompose(R); // return Cholesky factor for internal use
  }
  
  // Matrix exponential using scaling and truncated Taylor
  matrix unvectorized_matrix_exp_taylor(vector b, int K, int s){
    int M = to_int(sqrt(num_elements(b))); // for internal use I don't check if length of b is a square number
    matrix[M,M] B = to_matrix(b,M,M);
    matrix[M,M] I = diag_matrix(rep_vector(1.0,M));
    
    B = (B+B')/2; // To avoid 'A is not symmetric' problem due to floating error.
    real mu_B = sum(diagonal(B))/M;
    B = B - mu_B*I; // Reduce norm for better precision
    
    matrix[M,M] X = B/pow(2.0,s); // scaled matrix whose exponent will be approximated by Taylor
    matrix[M,M] EX = I; // exponent approximant starting from identity
    matrix[M,M] EX_increment = I;
    for(k in 1:K){
      EX_increment = EX_increment * X / k; // Taylor's expansion
      EX += EX_increment;
    }
    
    // scale EX back to B
    for(k in 1:s){
      EX = EX * EX;
    }
    EX = (EX + EX') / 2; // To avoid 'A is not symmetric' problem due to floating error.
    return exp(mu_B/2)*cholesky_decompose(EX); // return Cholesky factor for internal use
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
