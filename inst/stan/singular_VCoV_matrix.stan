  // Convert VCoV to standard deviation and correlation matrix
  tuple(vector,matrix) get_std_and_corr_of_VCoV(matrix VCoV){
    int M = cols(VCoV);
    vector[M] sigma = sqrt(diagonal(VCoV));
    return (sigma,quad_form_diag(VCoV, 1.0./sigma));
  }
  
  // transform matrix for a singular VCoV through quad_form()
  matrix sum_to_zero_VCoV_transform_matrix(int M){
    matrix[M,M+1] trans_M;
    trans_M[,1:M] = identity_matrix(M);
    trans_M[,M+1] = rep_vector(-1.0,M);
    return trans_M;
  }
  
  // full-rank VCoV with the least correlation under sum-to-zero constraint
  matrix equal_variance_sum_to_zero_VCoV(int M, int no_scale){
    matrix[M,M] eqvL = rep_matrix(-1.0,M,M);
    eqvL = add_diag(eqvL,1.0+M);
    if(no_scale) return eqvL;
    eqvL = eqvL/M;
    return eqvL;
  }
  
  // inverted VCoV matrix of above
  matrix inv_equal_variance_sum_to_zero_VCoV(int M, int no_scale){
    matrix[M,M] eqvL = rep_matrix(1.0,M,M);
    eqvL = add_diag(eqvL,1.0);
    if(no_scale) return eqvL;
    eqvL = (M*eqvL)/(M+1);
    return eqvL;
  }

  // I-J matrix as the product of canonical Helmert matrix
  matrix I_minus_J_matrix(int M){
    matrix[M,M] IJ = rep_matrix(-1.0/M,M,M);
    IJ = add_diag(IJ,1.0);
    return IJ;
  }
  