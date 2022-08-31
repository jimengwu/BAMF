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
  int<lower=0> m; // number of amus
  int<lower=0> O; // number of coefficients
  int<lower=0> n; // number of timepoints
  int<lower=0> p; // latent dimensions
  matrix<lower=0>[n,m] sigma; 
  matrix[n,O] B; // Bernstein basis
  // matrix[O,O] D; // Derivative operator
  matrix[n,m] Xv; // observations
  //matrix[T,T] C; // centering matrix
  //matrix[M,M] S; //Scaling matrix
   //matrix[P,M] F;
   vector[m] concentration;
}
parameters {
 matrix[O,p] A;
 positive_ordered[p] sigma_a;
 simplex[m] F[p]; 
 matrix[n,m] Sigma;
 
 real<lower=0> sigmaE;
}

transformed parameters{
  matrix[n,p] G; 
  //matrix[T,M] E;
  G = B * A + Sigma; 
  //matrix[T,M] G_hat;
  //E = C*(Sigma .* sigma)*S;
  // G_hat= B*A*vector_array_to_matrix(F);
}

model { 
  to_vector(Sigma) ~ normal(0,1);

  for(j in 1:O){
      A[j,:] ~ normal(0, sigma_a);
  }
  target += normal_lpdf(to_vector(Xv) | to_vector(G * vector_array_to_matrix(F)), 2*to_vector(sigma));
 
}

//generated quantities{
//  matrix[T,P] G;
//  G = (B*A);
//}