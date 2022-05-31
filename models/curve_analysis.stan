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
   matrix[P,M] F;
   vector[M] concentration;
}
parameters {
 matrix<lower=0>[O,P] A;
 real<lower=0> sigma_a[P];
 real<lower=0> sigmaE[P];
 #simplex[M] F[P]; 
#  real <lower=0> sigma;
 vector[P] tau;
 
}
transformed parameters{
    matrix[O,P] DG; 
    matrix[T,P] G;
    matrix[P,P] C;
   
    G = B*A;
    DG =  D * A ; 
  
  
   // for( i in 1:P){
  //    for(j in 1:P){
  //      if(i==j)
  //        C[i,j] = -exp(tau[i]);
  //      else
  //        C[i,j] = 0;
        
  //    }
      
  //  }
    
}

model { 
  #sigmaE~ scaled_inv_chi_square(1000,1);

  #target += normal_lpdf(to_vector(Xv) | to_vector(G* vector_array_to_matrix(F)), to_vector(sigma));
  target += normal_lpdf(to_vector(Xv) | to_vector(G* F), to_vector(sigma));
 
  //for (i in 1:P)
  //{
   // target += normal_lpdf(to_vector(DG) | to_vector(A*C), sigmaE[i]);
   # F[i] ~ dirichlet(concentration);
   // A[1,i] ~ normal(0,10);
    //for(j in 2:O){
    //  A[j,i] ~ normal(A[j-1,i], sigma_a[i]);
    //}
      
  //}
 
}
