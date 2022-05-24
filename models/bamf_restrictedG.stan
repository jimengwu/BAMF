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
  int<lower=1> n;     // number of data points
  int<lower=1> m;     // number of dimensions
  int<lower=1> p;     // number of components
  matrix[n,m] X;      // data matrix
  matrix<lower=0>[n,m] sigma;  // separate sigma for each of the observations (!)
  vector<lower=0>[n-1] timesteps; //How much time elapses between values in X (arbitrary unit, but unit choice also affects other priors)
  real<lower=0> minerr; //Minimal error
  
}

transformed data {
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);
  matrix[n,p] G_max = get_g_max(X,n,p);
}

parameters {
  matrix<lower=0,upper=1>[n,p] G_raw; // raw score matrix
  simplex[m] F[p]; // loadings simplex
  vector<lower=0>[p] alpha_a; // Posterior a
  vector<lower=minerr>[p] alpha_b; // Posterior b

}

transformed parameters {
  matrix[n,p] G = (G_max) .* G_raw; // limited score matrix
}

model {

  // The priors for G and F are implicitly given in the parameters section,
  //  so explicitly stating them here would only lead to unnecessary calculations
  //  (uniform priors become additive constants (in logspace), that can be ignored).
  
  // Column-wise autocorrelation
  for(j in 1:p) {
    // G[2:,j] ~ cauchy(G[:(n-1),j], alpha_a[j]*timesteps + alpha_b[j]);
    target += cauchy_lpdf(G[2:,j] | G[:(n-1),j], alpha_a[j]*timesteps + alpha_b[j]);
  }

  // X ~ Z = G * F
  // Xv ~ normal(to_vector(G*vector_array_to_matrix(F)), sigmav);
  target += normal_lpdf(Xv | to_vector(G*vector_array_to_matrix(F)), sigmav);

}
