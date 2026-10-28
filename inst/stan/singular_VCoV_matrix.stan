  // canonical Helmert matrix
  matrix canonical_Helmert(int M){
    matrix[M,M-1] H = rep_matrix(0,M,M-1);
    for(m in 1:(M-1)){
      H[1:m,m] = rep_vector(1/sqrt(m*(m+1)),m);
      H[m+1,m] = -m/sqrt(m*(m+1));
    }
    return H;
  }