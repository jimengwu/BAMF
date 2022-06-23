library(ggplot2)
library(ggfortify)
library(posterior)
library(bayesplot)
library(colormap)
library(corrplot)

c_light <- c("#DCBCBC")
c_light_highlight <- c("#C79999")
c_mid <- c("#B97C7C")
c_mid_highlight <- c("#A25050")
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")

nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C", "#A25050", "#8F2727", "#7C0000")

c_light_teal <- c("#6B8E8E")
c_mid_teal <- c("#487575")
c_dark_teal <- c("#1D4F4F")

summarise_diagnostics <- function(cmdstan_fit_object){
  cmdstan_fit_object$cmdstan_diagnose()
}
  
summarise_posterior <- function(cmdstan_fit_object){
  cmdstan_fit_object$summary()
}

plot_posterior_g <- function(cmdstan_fit_object, true_G, component_number_true, component_number_post){
  # convert to stan fit object
  stan_fit <- rstan::read_stan_csv(cmdstan_fit_object$output_files())
  
  probs = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  samples <- rstan::extract(stan_fit)
  true_component <- true_G[,component_number_true]
  x <- 1:length(true_component)
  cred <- sapply(1:length(x),
                 function(n) quantile(samples$G[,n,component_number_post], probs=probs))
  max_y <- max(true_component,  max(cred[9,]))
  min_y <- min(true_component, min(cred[1,]))
  plot(1, type="n", main=paste(
    c("G_true:",
      as.character(component_number_true),
      "G_post:",
      as.character(component_number_post)),
    collapse=" "
    ),
       xlab="x", ylab="f", xlim=c(min(x), max(x)), ylim=c(min_y, max_y))
  polygon(c(x, rev(x)), c(cred[1,], rev(cred[9,])),
          col = c_light, border = NA)
  polygon(c(x, rev(x)), c(cred[2,], rev(cred[8,])),
          col = c_light_highlight, border = NA)
  polygon(c(x, rev(x)), c(cred[3,], rev(cred[7,])),
          col = c_mid, border = NA)
  polygon(c(x, rev(x)), c(cred[4,], rev(cred[6,])),
          col = c_mid_highlight, border = NA)
  lines(x, cred[5,], col=c_dark, lwd=2)

  lines(x, true_component, lwd=4, xlab="x", ylab="f", col="white")
  lines(x, true_component, lwd=2, xlab="x", ylab="f", col="black")
  
  pp_check(as.array(true_component),samples$G[,,component_number_post],ppc_dens_overlay)
  #corrplot::corrplot(cor(,true_G))
}

plot_posterior_f <- function(cmdstan_fit_object, true_F, component_number_true, component_number_post){
  # convert to stan fit object
  stan_fit <- rstan::read_stan_csv(cmdstan_fit_object$output_files())
  
  probs = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  samples <- rstan::extract(stan_fit)
  true_component <- true_F[component_number_true,]
  x <- 1:length(true_component)
  cred <- sapply(1:length(x),
                 function(n) quantile(samples$F[,component_number_post,n], probs=probs))
  max_y <- max(true_component,  max(cred[9,]))
  min_y <- min(true_component, min(cred[1,]))
  plot(1, type="n", main=paste("F-",as.character(component_number_true)),
       xlab="x", ylab="f", xlim=c(min(x), max(x)), ylim=c(min_y, max_y))
  polygon(c(x, rev(x)), c(cred[1,], rev(cred[9,])),
          col = c_light, border = NA)
  polygon(c(x, rev(x)), c(cred[2,], rev(cred[8,])),
          col = c_light_highlight, border = NA)
  polygon(c(x, rev(x)), c(cred[3,], rev(cred[7,])),
          col = c_mid, border = NA)
  polygon(c(x, rev(x)), c(cred[4,], rev(cred[6,])),
          col = c_mid_highlight, border = NA)
  lines(x, cred[5,], col=c_dark, lwd=2)
  
  lines(x, true_component, lwd=4, xlab="x", ylab="f", col="white")
  lines(x, true_component, lwd=2, xlab="x", ylab="f", col="black")
  
}



