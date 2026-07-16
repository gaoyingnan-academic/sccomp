  // Convert VCoV to standard deviation and correlation matrix
  tuple(row_vector,matrix) get_std_and_corr_of_VCoV(matrix VCoV){
    int M = cols(VCoV);
    vector[M] sigma = sqrt(diagonal(VCoV));
    return (to_row_vector(sigma),quad_form_diag(VCoV, 1.0./sigma));
  }
  
  // canonical Helmert matrix
  matrix canonical_Helmert(int M){
    matrix[M,M-1] H = rep_matrix(0,M,M-1);
    for(m in 1:(M-1)){
      H[1:m,m] = rep_vector(-1/sqrt(m*(m+1)),m);
      H[m+1,m] = m/sqrt(m*(m+1));
    }
    return H;
  }