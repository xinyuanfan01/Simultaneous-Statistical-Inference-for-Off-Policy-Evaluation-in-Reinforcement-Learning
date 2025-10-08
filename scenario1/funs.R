library(Rcpp)
library(ggplot2)
library(RCurl)
library(dplyr) 
library(foreach)
library(parallel)
library(doParallel)
library(doSNOW)
library(splines) 


sigmoid<-function(x){
  1/(1+exp(-x))
}

target_policy<-function(x){
  1-all(x>0)
}



action<-function(x){ 
  p=0.5
  rbinom(1,1,p)
}



new_state<-function(x,a){
  x +(2*a-1)*runif(1,0,1)
  
}

reward<-function(x,a){
  
  0.5*tanh(-x^2)
  
}


transform_state<-function(x){
  
  sigmoid(x)
  
}


V<-function(x,Tot,gamma,B=1000,dimen=1,reward_1=reward,state=new_state,policy=target_policy){
  res=rep(0,B)
  for(b in 1:B){
    X_i=array(0,dim=c(dimen,Tot))
    A_i=rep(0,Tot)
    Y_i=rep(0,Tot)
    X_i[,1]=x
    for(t in 1:(Tot-1)){
      A_i[t]=policy(X_i[,t])
      X_i[,t+1]=state(X_i[,t],A_i[t])
    }
    A_i[Tot]=policy(X_i[,Tot])
    for(t in 1:(Tot-1)){
      Y_i[t]=reward(X_i[,t+1],A_i[t])
    }
    Y_i[Tot]=reward(new_state(X_i[,Tot],A_i[Tot]),A_i[Tot])
    
    res[b]=sum(Y_i*gamma^(0:(Tot-1)))
  }
  mean(res)
}



hat_V<-function(x,hat_beta,target_policy,...){
  as.numeric(t(U(x,target_policy,...))%*%hat_beta)
}

xi<-function(x,a){
  mat=rep(0,L*m)
  
  mat[a*L+1:L]=get_phi(x)##phi_L(x)##
  matrix(mat)
}

U<-function(x,target_policy,...){
  a=target_policy(x,...)
  mat=rep(0,L*m)
  mat[a*L+1:L]=get_phi(x)#
  #phi_L(x)#
  matrix(mat)
}

phi_L<-function(x,dimen=1){
  x=transform_state(x)
  xx = seq(0, 1, length.out = 1000)
  if(dimen==2){
    s1=purrr::map_dbl(1:(L1),~approx(xx, basis_mat[,.], xout = x[1])$y)
    s2=purrr::map_dbl(1:(L1),~approx(xx, basis_mat[,.], xout = x[2])$y)
    return(as.vector(
      outer(
        s1,s2
      )
    ))
  }else{
    
    a=purrr::map_dbl(1:(L1),~approx(xx, basis_mat[,.], xout = x[1])$y)
    return(a )
  }
}


estimated_V<-function(X,A,Y,n,gamma,target_policy,...){
  hat_Sigma_pi=0
  s=0
  for(ii in 1:n){
    for(t in 1:(length(Y[[ii]])-1)){
      xit=xi(X[[ii]][,t],A[[ii]][t])
      Uit1=U(X[[ii]][,t+1],target_policy,...)
      if(any(is.nan(xit))){stop("xit")}
      if(any(is.nan(Uit1))){stop("Uit1")}
      #,...)
      hat_Sigma_pi=hat_Sigma_pi+xit%*%t(xit-gamma*Uit1)
      s=s+xit*Y[[ii]][t]
      if(any(is.nan(s))){stop("s")}
    }
  } 
  sumTi=sum(unlist(lapply(Y,length))-1)
  hat_Sigma_pi=hat_Sigma_pi/sumTi
  s=s/sumTi
  
  hat_beta=solve(hat_Sigma_pi)%*%s
  
  #CI
  Omega_pi=0
  for(ii in 1:n){
    for(t in 1:(length(Y[[ii]])-1)){
      xit=xi(X[[ii]][,t],A[[ii]][t])
      Uit1=U(X[[ii]][,t+1],target_policy,...)
      Omega_pi=Omega_pi+xit%*%t(xit)*
        as.numeric((Y[[ii]][t]+gamma*t(Uit1)%*%hat_beta-
                      t( get_phi(X[[ii]][,t]))%*%hat_beta[A[[ii]][t]*L+1:L]
        )^2)
    }
  }
  Omega_pi=Omega_pi/sumTi
  
  inv_hat_Sigma_pi=solve(hat_Sigma_pi)
  cov_mat=inv_hat_Sigma_pi%*%Omega_pi%*%t(inv_hat_Sigma_pi)
  return(list(hat_beta,cov_mat,sumTi,inv_hat_Sigma_pi))
}

bootstrap_quantile<-function(state_grid,cov_mat,target_policy,addi,Bn=1e4,...){
  boot_normal_rv=t(MASS::mvrnorm(Bn,rep(0,nrow(cov_mat)),cov_mat))
  
  a1=purrr::map(1:ncol(state_grid),function(ii){
    xx=state_grid[,ii]
    as.numeric(
      abs(
        t(U(xx,target_policy,...))%*%boot_normal_rv
      ) 
    )
  })
  
  c(NA,quantile(do.call(pmax, a1) ,0.95)
  )
}



generate_legendre_basis <- function(order, grid_size = 1000) { 
  x <- seq(0, 1, length.out = grid_size)
  x_mapped <- 2 * x - 1   
  A=orthopolynom::legendre.polynomials(order,T)
  A_mat <- sapply(A, function(p) predict(p, x_mapped))
  for(i in 1:ncol(A_mat)){
    A_mat[,i]=A_mat[,i]/sqrt(0.5)
  }
  return(A_mat)
}



generate_grid_from <- function(mat_list, n_points = 10) {
  range1 <- mat_list
  range2 <- mat_list
  grid1 <- seq(range1[1], range1[2], length.out = n_points)
  grid2 <- seq(range2[1], range2[2], length.out = n_points)
  grid <- expand.grid(grid1, grid2)
  grid_matrix <- t(as.matrix(grid))  # 
  return(grid_matrix)
}



