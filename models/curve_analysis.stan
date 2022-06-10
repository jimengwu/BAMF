functions {
  matrix vector_array_to_matrix(vector[] x) {
    matrix[size(x), rows(x[1])] y;
    for (m in 1:size(x))
      y[m] = x[m]';
    return y;
  }
  
  # limit the G value by the sum of X matrix on row, the total concentration of every measured chemical 
  # should be larger than the G value at time point t. 
  matrix get_g_max(matrix x,int n,int p) {
    matrix[n,p] y;
    for (i in 1:n)
      for (j in 1:p)
        y[i][j] = sum(x[i])*2;
    return y;
  }
  
}

data {
  int<lower=0> N; #number of series
  int<lower=0> M; # number of amus
  int<lower=0> O; #number of coefficients
  int<lower=0> T; #number of timepoints
  int<lower=0> P; #latent dimensions
  matrix<lower=0>[T,M] sigma; 
  matrix<lower=0,upper=1>[T,O] B; # Bernstein basis
  matrix[O,O] D; # Derivative operator
  matrix[T,M] Xv; #observations
  real<lower=0> MAX;
   //matrix[P,M] F;
   vector[M] concentration;
}
parameters {
 matrix<lower=0>[O,P] A;
 ordered[P] sigma_a;
 ordered[P] sigmaE;
 real alpha[P];
 real beta[P];
 simplex[M] F[P]; 
#  real <lower=0> sigma;
 ordered[P] tau;
 matrix[T,P] G;
// cov_matrix[P] C;
 real gamma[P];
}
transformed parameters{
    matrix[O,P] DG; 
    matrix[O,P] AC;
    
   
    DG =  D * A ; 
  //  AC = A * C;
  
  
   
    
}

model { 
  #sigmaE~ scaled_inv_chi_square(1000,1);
  for( i in 1:P){
    G[:,i] ~ normal(B*A[,i], exp(sigmaE[P]));
    sigma_a[i] ~ normal(0, 10^i);
    //target += normal_lpdf(DG[:,i] | AC[:,i] , exp(tau[i]) );
  }
  
  // 
  target += normal_lpdf(to_vector(Xv) | to_vector(G* vector_array_to_matrix(F)),  to_vector(sigma));
  // sigmaE ~ normal(0,1);
  for(j in 1:O){
      
      A[j,:] ~ normal(0, exp(sigma_a));
  }
  
 
}
