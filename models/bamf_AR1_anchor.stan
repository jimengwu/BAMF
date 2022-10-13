functions {
  matrix vector_array_to_matrix(vector[] x) {
    matrix[size(x), rows(x[1])] y;
    for (m in 1:size(x))
      y[m] = x[m]';
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
  
  int<lower=1>S;
  int selected_indexes[S];
  
    
  matrix<lower=0>[p,S] F;
  int not_selected_indexes[m-S];
}


transformed data {
  vector[n*S] Xv = to_vector(X[:,selected_indexes]);
  vector[n*S] sigmav = to_vector(sigma[:,selected_indexes]);
   
     
  vector[n*(m-S)] Xn = to_vector(X[:,not_selected_indexes]);
  vector[n*(m-S)] sigma_n = to_vector(sigma[:,not_selected_indexes]);
  
  vector[p] row_sums;
  
  for(i in 1:p){
    row_sums[i]=sum(F[i,:]);
  }
  
  
}

parameters {
  matrix<lower=0>[n,p] G; // score matrix
  vector<lower=0>[p] alpha_a; // Posterior a
  vector<lower=minerr>[p] alpha_b; // Posterior b
  
  simplex[m-S] F_n[p];
  real<lower=0> sigma_a[p];
}

transformed parameters{
  matrix[p,m-S] F_non_selected;
  for(i in 1:p){
    F_non_selected[i,:] = (1-row_sums[i])*F_n[i]';
  }
}


model {

  // The priors for G and F are implicitly given in the parameters section,
  //  so explicitly stating them here would only lead to unnecessary calculations
  //  (uniform priors become additive constants (in logspace), that can be ignored).
  
  // Column-wise autocorrelation
  for(j in 1:p) {
    G[1,j] ~ normal(0,100); // the first time point regularised
    G[2:,j] ~ normal(alpha_a[j] + alpha_b[j]*G[:(n-1),j], sigma_a[j]); // gaussian
  }

  // X ~ Z = G * F
  Xv ~ normal(to_vector(G*F), sigmav);
  Xn ~ normal(to_vector(G*F_non_selected), sigma_n);
}