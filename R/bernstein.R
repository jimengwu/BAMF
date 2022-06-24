bernstein.basis <- function(m,t,a=NULL,b=NULL){
  a<-ifelse(is.null(a),0,a)
  b<-ifelse( is.null(b),1,b)
  k=1:m
  choose(m-1,k-1)*(((t-a)^(k-1))*((b-t)^(m-k)))/((b-a)^(m-1))
}

bernstein.matrix <- function(m,time,a,b){
  sapply(X = time,FUN=function(x)bernstein.basis(m,x,a,b))
}

bernstein.basis.derivative<- function(m,t){
  (m-1)*bernstein.basis(m-1,t)
}

bernstein.basis.derivative.matrix<- function(m,time){
  sapply(X = time,FUN=function(x)bernstein.basis.derivative(m,x))
}


normalize.interval <- function(x,a=x[1],b=x[length(x)]){
  (x-a)/(b-a)
}

derivative.coefficients <- function(coefficients,order=1){
  M<-length(coefficients)
  if(order==1){
    (M-1)*(coefficients[2:length(coefficients)]-coefficients[1:(length(coefficients)-1)])
  }
  else{
    coef<-(coefficients[2:length(coefficients)]-coefficients[1:(length(coefficients)-1)])
    (M-1)*(derivative.coefficients(coef,order=order-1))
  }
}



get.roots<- function(coefficients){
  L <- length(coefficients)
  companion<- as.matrix(cbind(diag(L-2),-coefficients[2:(L-1)]))
  companion <- rbind(c(rep(0,L-2),-coefficients[1]),companion)
  eigen(companion)$values
}

coef.bernstein.to.power <- function(x){
  m <- length(x)
  k <- 1:m
  result<-x*choose(m-1,k-1)
  #the companion matrix requires the first coefficient to be 1
  if(result[m]!=0)
    result <- result/result[m]
  else{
    result <- result[1:m-1]
    result <- result/result[m-1]
  }
  result
}


get.maximum.values <- function(x){
  L<-length(x)
  m <- L
  dx <- derivative.coefficients(x) 
  cdx<-coef.bernstein.to.power(dx)
  tmp.t<-get.roots(cdx)
  tmp.t<-tmp.t/(1+tmp.t)
  tmp.t <- tmp.t[abs(Im(tmp.t)) < 1e-6]
  tmp.t<- Re(tmp.t)
  time<-tmp.t
  X <-BernsteinMatrix(m,time)
  tmp.v<-x%*%X
  
  value<-t(tmp.v)
  
  result<- cbind(t(t(as.array(time))),value,deparse.level = 0)
  result<-result[result[,2]==max(value),]
  result
}


bernstein.derivative.operator<- function(m){
  n.row = m-1
  n.col = m
  D <- matrix(nrow = n.row,ncol = n.col)
  for(i in 1:n.row){
    for(j in 1:n.col){
      if(i==j){
        D[i,j] <- -1
      }else{
        if(j==(i+1)){
          D[i,j] <- 1
        }else{
          D[i,j] <- 0
        }
        
      }
      
    }
  }
  L <- matrix(ncol=n.col,nrow=n.row)
  for(i in 1:n.row){
    for(j in 1:n.col){
      L[i,j]=0
      if(i==j){
        L[i,j]=((n.col-1)-(i-1))/(n.col-1)
      }
      if(j==(i+1)){
        L[i,j]=(j-1)/(n.col-1)  
      }
    }
  }
  
  (m-1)*t(D)%*%L
}

bernstein.comb.matrix<-function(M){
  Aj<-vector(mode = "list",length=M)
  for(i in 0:(M-1)){
    Ai=c(rep(0,i))
    for(j in 0: (M-1-i))
      Ai=c(Ai,((-1)^(M-j))*choose(M-1,i)*choose(M-1-i,j))
    Aj[[i+1]]=Ai
  }
  matrix(unlist(Aj),M,M,byrow = T)
}

bernstein.lambda.matrix<-function(M){
  diag(1/1:M)
}


bernstein.integral.operator<-function(M){
  H<-Hilbert(M)
  A<-bernstein.comb.matrix(M)
  Q<-A%*%H%*%t(A)
  Ainv<-solve(A)
  cm=0:(M-1)
  cm<-choose(M-1,cm)
  cdm<-1:((M-1)+1)
  cdm<-choose(2*(M-1)+1,(M-1)+cdm)
  cm1<-cm/cdm
  Qinv<-solve(Q)
  cm1<-(Qinv/(2*(M-1)+2))%*%as.matrix(cm1)
  B<-matrix(nrow = M,ncol=M)
  for(i in 2:M)
    B[i-1,]<-Ainv[i,]
  B[M,]<-as.vector(cm1)
  lambda<-bernstein.lambda.matrix(M)
  as.matrix(A%*%lambda%*%B)
}

subdivision.matrix<-function(cut,M){
  A<-bernstein.comb.matrix(M)
  d<-cut^(0:(M-1))
  S<-diag(d)
  tmp<-solve(A,S)
  A%*%S%*%solve(A)
}

bernstein.derivative.operator2<- function(M){
  lambda<-rbind(rep(0,M-1),diag(1:(M-1)))
  A<-bernstein.comb.matrix(M)
  Ainv<-solve(A)
  B<-matrix(nrow=M-1,ncol=M)
  for(i in 1:(M-1)){
    B[i,]<-Ainv[i,]
  }
  A%*%lambda%*%B  
}

bernstein.product.vector<-function(k,i,M){
  H<-Hilbert(M);
  A<-bernstein.comb.matrix(M)
  Q<-A%*%H%*%t(A)
  Qinv<-solve(Q)
  const<-choose(M-1,i)*Qinv/(2*(M-1)+k+1)
  numerator<-0:(M-1)
  denominator<-0:(M-1)
  numerator<-choose(M-1,numerator)
  denominator<-choose(2*(M-1)+k,i+k+denominator)
  vector<-numerator/denominator
  const%*%as.matrix(vector)
}
