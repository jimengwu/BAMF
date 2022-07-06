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

extract_cmdstan_samples <- function(cmdstan_fit_object){
  stan_fit <- rstan::read_stan_csv(cmdstan_fit_object$output_files())
  samples <- rstan::extract(stan_fit)
}


plot_posterior_g <- function(samples, true_G, component_number_true,
                             component_number_post){
  # convert to stan fit object
  par(mar = c(3, 3, 2, 1), # Dist' from plot to side of page
      mgp = c(2, 0.4, 0), # Dist' plot to label
      las = 1, # Rotate y-axis text
      tck = -.01, # Reduce tick length
      xaxs = "i", yaxs = "i") # Remove plot padding 
  probs = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  
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
  #corrplot::corrplot(cor(,true_G))
}

plot_z_scores_g <- function(samples, true_G, component_number_true, component_number_post){
  alpha = 150 
  palette(c(rgb(200, 79, 178, alpha = alpha, maxColorValue = 255),
            rgb(105, 147, 45, alpha = alpha, maxColorValue = 255),
            rgb(85, 130, 169, alpha = alpha, maxColorValue = 255)))
  y_hat = samples$G[,,component_number_post] %>% colMeans()
  y_sd = samples$G[,,component_number_post] %>% matrixStats::colSds()
  y = as.matrix(G_)[,component_number_true]
  z_score = (y-y_hat)/y_sd
  
  par(mar = c(3, 3, 2, 1), # Dist' from plot to side of page
      mgp = c(2, 0.4, 0), # Dist' plot to label
      las = 1, # Rotate y-axis text
      tck = -.01, # Reduce tick length
      xaxs = "i", yaxs = "i") # Remove plot padding
  plot(z_score, 
       bg = ifelse(abs(z_score)>2,c_dark,c_light_teal),
       pch = 21,
       frame.plot=F)
  abline(h=2)
  abline(h=-2)
}

plot_mean_error_g <- function(samples, true_G,component_number_true, component_number_post){
  y_hat = samples$G[,,component_number_post] %>% colMeans()
  y_true = as.matrix(true_G)[,component_number_true]
  alpha = 150 
  palette(c(rgb(200, 79, 178, alpha = alpha, maxColorValue = 255),
            rgb(105, 147, 45, alpha = alpha, maxColorValue = 255),
            rgb(85, 130, 169, alpha = alpha, maxColorValue = 255)))
  par(mar = c(3, 3, 2, 1), # Dist' from plot to side of page
      mgp = c(2, 0.4, 0), # Dist' plot to label
      las = 1, # Rotate y-axis text
      tck = -.01, # Reduce tick length
      xaxs = "i", yaxs = "i") # Remove plot padding
  plot(lm(y_true~y_hat))
  plot(y_true ~ y_hat)
  abline(lm(y_true~y_hat))
  abline(0,1,col="red")
}
  
plot_median_error_g <- function(samples, true_G,component_number_true, component_number_post){
  y_hat = samples$G[,,component_number_post] %>% matrixStats::colMedians() 
  y_true = as.matrix(true_G)[,component_number_true]
  alpha = 150 
  palette(c(rgb(200, 79, 178, alpha = alpha, maxColorValue = 255),
            rgb(105, 147, 45, alpha = alpha, maxColorValue = 255),
            rgb(85, 130, 169, alpha = alpha, maxColorValue = 255)))
  par(mar = c(3, 3, 2, 1), # Dist' from plot to side of page
      mgp = c(2, 0.4, 0), # Dist' plot to label
      las = 1, # Rotate y-axis text
      tck = -.01, # Reduce tick length
      xaxs = "i", yaxs = "i") # Remove plot padding
  plot(lm(y_true~y_hat))
  plot(y_true ~ y_hat)
  abline(lm(y_true~y_hat))
  abline(0,1, col="red")
}

cor_plot_g <- function(samples, true_G, func="mean"){
  if(func=="mean"){
    post_G <- apply(samples$G, MARGIN=c(2,3), mean)
  }
  if(func=="median"){
     post_G <- apply(samples$G, MARGIN=c(2,3), median)
  }
  true_G <- as.matrix(true_G)
  corrplot(cor(post_G, true_G))
  return(cor(post_G,true_G))
}
  
get_labels_g <- function(cor_matrix){
  trues <- apply(cor_matrix,MARGIN=1, which.max)
  cbind(1:nrow(cor_matrix),trues)
}

posterior_analysis <- function(csv_file_posterior, G_, output_file){
  P <- ncol(G_)
  fit_mcmc <- cmdstanr::as_cmdstan_fit(csv_file_posterior)
  

  samples <-  extract_cmdstan_samples(fit_mcmc)

  pdf(output_file)

  mean_cor<- cor_plot_g(samples, G_)
  median_cor <- cor_plot_g(samples,G_,"median")
  apply(mean_cor, MARGIN = 1, FUN = which.max) 

  mean_labels = get_labels_g(cor_matrix = mean_cor)
  get_labels_g(cor_matrix = median_cor)
  
  for(i in 1:P){
    plot_posterior_g(samples=samples,true_G = as.matrix(G_), component_number_true = mean_labels[i,2] ,component_number_post = i )
      x = list( y = as.matrix(G_)[,mean_labels[i,2]], y_rep= samples$G[,,i])
      print(bayesplot::pp_check(x$y , x$y_rep, ppc_dens_overlay))
      plot_z_scores_g(samples, G_, mean_labels[i,2], i)
      plot_median_error_g(samples, G_, mean_labels[i,2], i)
     
      #blue <- rgb(0, 0, 1, alpha=0.1)
      #n<- barplot( as.numeric(F_[i,]),col=blue, main=paste0("F component:", as.character(i))) 
      #boxplot(samples$F[,i,], frame=F,range=0, at=n, add=T)
  }


  print(bayesplot::mcmc_trace(fit_mcmc$draws(),regex_pars = "alpha"))

  dev.off()
}


map_analysis <- function(fit_map,G_,output_file){
  P <- ncol(G_)
  N <- nrow(G_)
  fit_map <- fit_bamf_gauss_noF_map
  draws <- fit_map$draws(format="df")
  result <- matrix(rep(0,P*N),ncol=P) 
  for(i in 1:P){
    select_string <- paste(c(",",as.character(i)),collapse="")
    current_column <- draws_ %>% as_tibble() %>% select(starts_with("G")) %>% 
      select(contains(select_string)) %>% as.numeric()
    current_column
    result[,i] <- current_column
  } 
  map_cor <- corrplot(cor(result,G_))
  mean_labels = get_labels_g(cor_matrix = cor(result,G_))
  for(i in 1:P){
     component_number_true <-mean_labels[i,2]
    component_number_post <- i
    data_map <- data.frame(true_G = as.matrix(G_)[, mean_labels[i,2]], map =result[, mean_labels[i,2]])
    plot(lm(true_G ~ map, data=data_map))
    y_hat = data_map$map
    y_true = data_map$true_G
    alpha = 150 
    palette(c(rgb(200, 79, 178, alpha = alpha, maxColorValue = 255),
              rgb(105, 147, 45, alpha = alpha, maxColorValue = 255),
              rgb(85, 130, 169, alpha = alpha, maxColorValue = 255)))
    par(mar = c(3, 3, 2, 1), # Dist' from plot to side of page
        mgp = c(2, 0.4, 0), # Dist' plot to label
        las = 1, # Rotate y-axis text
        tck = -.01, # Reduce tick length
        xaxs = "i", yaxs = "i") # Remove plot padding
    plot(y_true ~ y_hat)
    abline(lm(y_true~y_hat))
    abline(0,1, col="red")
    
    true_component <- y_true
    x <- 1:length(true_component)
   
    max_y <- max(true_component)
    min_y <- min(true_component)
    plot(1, type="n", main=paste(
      c("G_true:",
        as.character(component_number_true),
        "G_post:",
        as.character(component_number_post)),
      collapse=" "
    ),
    xlab="x", ylab="f", xlim=c(min(x), max(x)), ylim=c(min_y, max_y))
    
    lines(x, y_hat, col=c_dark, lwd=2)
    lines(x, true_component, lwd=4, xlab="x", ylab="f", col="white")
    lines(x, true_component, lwd=2, xlab="x", ylab="f", col="black")
    #  plot_mean_error_g(samples, G_, mean_labels[i,2], i)
  }
}