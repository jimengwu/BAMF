G_ = read.csv(file = "../data/example_data_G.csv",header=TRUE)
F_ = read.csv(file = "../data/example_data_F.csv",header=TRUE)
E_ = read.csv(file = "../data/example_data_E.csv", header=TRUE)
X_ = read.csv(file = "../data/example_data_X.csv", header=TRUE)

N <- nrow(G_[,-1])
M <- ncol(X_[,-1])
P <- ncol(G_[,-1])
t <- G_[,1]
t <- (t - t[1])/(1e9)/3600
t<- diff(t)
X_ <- X_[,-1]
E_ <- E_[,-1]
F_ <- F_[,-1]
G_ <- G_[,-1]
F_ <- apply(F_, MARGIN=2, FUN = as.numeric)
par(mfrow=c(2,3))
plot(G_[,1], type="l")
plot(G_[,2], type="l")
plot(G_[,3], type="l")
plot(F_[1,], type="l")
plot(F_[2,], type="l")
plot(F_[3,], type="l")


c <-1
G_0 <- function(n) rbeta(n, 2, 5)
G_0 <- function(n) rnorm(39,19/39, 0.5/39)
G_0 <- function(n) F_[1,]
n <- 39
b <- rbeta(n, 1, c)
p <- numeric(n)
p[1] <- b[1]
p[2:n] <- sapply(2:n, function(i) b[i] * prod(1 - b[1:(i-1)]))
y <- G_0(n)
theta <- sample(y, prob = p, replace = TRUE)
par(mfrow=c(3,3))
hist(y,breaks=39)
hist(theta,breaks=39)
hist(p,breaks=39)
plot(F_[1,],p)
plot(F_[1,])
plot(p)
plot(exp(theta)/sum(exp(theta)))
quantile(theta, na.rm = FALSE,
         names = TRUE)
