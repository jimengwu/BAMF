functions {
  matrix vector_array_to_matrix(vector[] x) {
    matrix[size(x), rows(x[1])] y;
    for (m in 1:size(x))
      y[m] = x[m]';
    return y;
  }
  
  matrix get_g_max(matrix x,int n,int p) {
    matrix[n,p] y;
    for (i in 1:n)
      for (j in 1:p)
        y[i][j] = sum(x[i])*2;
    return y;
  }
  
}

data {
  int<lower=0> N; // number of series
  int<lower=0> M; // number of amus
  int<lower=0> O; // number of coefficients
  int<lower=0> T; // number of timepoints
  int<lower=0> P; // latent dimensions
  matrix<lower=0>[T,M] sigma; 
  matrix[T,O] B; // Bernstein basis
  // matrix[O,O] D; // Derivative operator
  matrix[T,M] Xv; // observations
  //matrix[T,T] C; // centering matrix
  //matrix[M,M] S; //Scaling matrix
  real<lower=0> MAX;
   //matrix[P,M] F;
   vector[M] concentration;
}
parameters {
 matrix[O,P] A;
 ordered[P] sigma_a;
 simplex[M] F[P]; 
 //matrix[T,M] Sigma;
 
 real<lower=0> sigmaE;
}

transformed parameters{
  
  //matrix[T,M] E;
  matrix[T,P] G;
  //matrix[T,M] G_hat;
  //E = C*(Sigma .* sigma)*S;
  G = B*A;
  // G_hat= B*A*vector_array_to_matrix(F);
}

model { 
  //to_vector(Sigma) ~ normal(0,1);
  for(j in 1:O){
      A[j,:] ~ normal(0, exp(sigma_a));
  }
  target += normal_lpdf(to_vector(Xv) | to_vector(G * vector_array_to_matrix(F)), to_vector(sigma));
 
}

//generated quantities{
//  matrix[T,P] G;
//  G = (B*A);
//}