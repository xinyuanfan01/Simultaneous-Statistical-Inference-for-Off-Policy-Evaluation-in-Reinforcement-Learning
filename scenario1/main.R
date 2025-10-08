# This file replicates Figure 1 presented in the main paper

source("funs.R") 
library(RhpcBLASctl) 

param_grid <- data.frame(n = c(25,25,50,25,75,50,50,75,75),
                         Tot = c(3,5,3,7,3,5,7,5,7)*10,
                         L=c(13,12,14,13,20,24,24,29,25)) 

no_cores <-detectCores()
cl <- makeCluster(no_cores)
registerDoParallel(cl)
registerDoSNOW(cl)
clusterEvalQ(cl,{library(RhpcBLASctl) 
  blas_set_num_threads(1) 
  source("funs.R") })

load("V_real_sequence_s1_1e4.rdata")
for(iii in 1:nrow(param_grid)){
  n=param_grid$n[iii]
  Tot=param_grid$Tot[iii]
  gamma=0.5
  dimen=1
  #points for evalution
  state_grid=matrix(seq(-2,2,length=1000),nrow=1)
  num_for_Bonferroni=ncol(state_grid)
  L1 =param_grid$L[iii]
  m=2 # action space: 0,...,m-1
  L=L1
  
  
  repeat_time=500
  
  #parallel 
  res_para <- foreach(kk_new=1:repeat_time,.combine='rbind')%dopar%{
    tryCatch({
      #generate data
      X <- vector("list", length = n)
      A <- vector("list", length = n)
      Y <- vector("list", length = n)
      for(ii in 1:n){
        X_i=array(0,dim=c(dimen,Tot))
        A_i=rep(0,Tot)
        Y_i=rep(0,Tot)
        X_i[,1]=runif(1,-2,2) 
        for(t in 1:(Tot-1)){
          A_i[t]=action(X_i[,t])
          X_i[,t+1]=new_state(X_i[,t],A_i[t])
        }
        A_i[Tot]=action(X_i[,Tot])
        for(t in 1:(Tot-1)){
          Y_i[t]=reward(X_i[,t+1],A_i[t])
        }
        Y_i[Tot]=reward(new_state(X_i[,Tot],A_i[Tot]),A_i[Tot])
        
        X[[ii]]=X_i
        A[[ii]]=A_i
        Y[[ii]]=Y_i
      }
      
      phi_cache <- new.env()
      get_phi <- function(x) {
        key <- paste0(round(x, 5), collapse = "_")
        if (!exists(key, envir = phi_cache)) {
          phi_cache[[key]] <- phi_L(x)
        }
        phi_cache[[key]]
      }
      
      
      basis_mat=generate_legendre_basis(L1-1) 
      
      #sieve
      est_tmp=estimated_V(X,A,Y,n,gamma,target_policy)
      hat_beta=est_tmp[[1]]
      cov_mat=est_tmp[[2]]
      sumTi=est_tmp[[3]]
      
      
      #SCB
      hat_V_sequence=purrr::map_dbl(1:ncol(state_grid),
                                    ~hat_V(state_grid[,.],hat_beta,target_policy))
      hat_sigma_sequence=purrr::map_dbl(1:ncol(state_grid),
                                        ~t(U(state_grid[,.],target_policy))%*%cov_mat%*%U(state_grid[,.],target_policy))/sumTi
      a_ttmp=mean(hat_sigma_sequence)*sumTi
      hat_sigma_sequence_SCB=purrr::map_dbl(1:ncol(state_grid),
                                            ~ (
                                              max(a_ttmp,t(U(state_grid[,.],target_policy))%*%
                                                    cov_mat%*%
                                                    U(state_grid[,.],target_policy))))/sumTi
      
      q1=bootstrap_quantile(state_grid,cov_mat,target_policy,addi=addi) 
      
      
      sd_sequence_equal=q1[2]*sqrt(1/sumTi)
      lb_sequence_equal=hat_V_sequence-sd_sequence_equal
      ub_sequence_equal=hat_V_sequence+sd_sequence_equal
      flag_sequence_equal=(lb_sequence_equal<V_real_sequence)&
        (ub_sequence_equal>V_real_sequence)
      flag_scb_equal=all(flag_sequence_equal)
      
      #SAVE method
      q2=qnorm(1-0.025/num_for_Bonferroni)
      #Sidak
      #q2=qnorm(1-(1-(1-0.05)^(1/num_for_Bonferroni))/2)
      
      sd_shi=q2*sqrt(hat_sigma_sequence)
      lb_shi=hat_V_sequence-sd_shi
      ub_shi=hat_V_sequence+sd_shi
      flag_shi=(lb_shi<V_real_sequence)&(ub_shi>V_real_sequence)
      flag_shi=all(flag_shi)
      
      c(NA,flag_shi, q1,
        hat_V_sequence,rep(NA,ncol(state_grid)),sd_shi, 
        flag_scb_equal,sd_sequence_equal)
      
    }, error = function(e) {
      message("Error in iteration ", kk_new, ": ", e$message) 
      return(rep(-1, 6 + 3 * ncol(state_grid)))
    })
  }
  
  data=data.frame(x=as.numeric(state_grid),
                  V_real_sequence1=V_real_sequence,
                  V_real_sequence=colMeans(res_para[,5:(5+num_for_Bonferroni-1)]),
                  lb_SCB=colMeans(res_para[,5:(5+num_for_Bonferroni-1)]-
                                    res_para[,(5+num_for_Bonferroni):(5+2*num_for_Bonferroni-1)]),
                  ub_SCB=colMeans(res_para[,5:(5+num_for_Bonferroni-1)]+
                                    res_para[,(5+num_for_Bonferroni):(5+2*num_for_Bonferroni-1)]),
                  lb_shi=colMeans(res_para[,5:(5+num_for_Bonferroni-1)]-
                                    res_para[,(5+2*num_for_Bonferroni):(5+3*num_for_Bonferroni-1)]),
                  ub_shi=colMeans(res_para[,5:(5+num_for_Bonferroni-1)]+
                                    res_para[,(5+2*num_for_Bonferroni):(5+3*num_for_Bonferroni-1)])
  )
  param_grid[iii,4:9]=c(colMeans(res_para[,c(1,ncol(res_para)-1,2)]),
                        c(mean(data$ub_SCB-data$lb_SCB),
                          mean(2*res_para[,ncol(res_para)]),
                          mean(data$ub_shi-data$lb_shi))
  )
  
  
}
stopCluster(cl)  

param_grid[,c(1,2,5,8,6,9)]
