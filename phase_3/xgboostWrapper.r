library(xgboost)
library(pROC)
library(doParallel)
library(foreach)
source('setup_crossval.R')
#Parameters
#Tree Depth 5 to 10
#Eta 0.1 to 0.8
#Thread all of them
#Iterations 500-1000

#TODO: Type.measure, parallel

xgcv <- function(train.data, train.labels, tree.depth, eta, iters, verbose, nfolds = 5) {
  #print(tree.depth)
  #print(eta)
  #print(iters)
  #run internal cross validation to determine the best via a grid search
  #expects a vector for tree.depth, eta, and iters
  if(verbose == TRUE) {
    cat(paste('xgcv: setting up internal cv\n'))
  }

  int.crossval.data <- crossval.list(train.data, train.labels, nfolds)
  # #  folds.lists <- list()
  # folds.lists[[1]] <- train.data.folds
  # folds.lists[[2]] <- test.data.folds
  # folds.lists[[3]] <- train.labels.folds
  # folds.lists[[4]] <- test.labels.folds
  # folds.lists[[5]] <- idx.pos
  # folds.lists[[6]] <- idx.neg
  # folds.lists[[7]] <- positives
  # folds.lists[[8]] <- negatives
  
  if(verbose == TRUE) {
    cat(paste('xgcv: finished internal cv\n'))
  }
  
  best.auc <- -Inf
  best.tr <- NA
  best.eta <- NA
  best.iter <- NA
  
  for(tr in tree.depth) {
    for(et in eta) {
      #for(it in iters) {

          xgb.auc <- vector()
          
        if(verbose == TRUE) {
          cat(paste('-----------------------\n'))
          cat(paste('xgcv: depth:', tr,'\n'))
          cat(paste('xgcv: eta:', et,'\n'))
          cat(paste('xgcv: iter:', max(iters),'\n'))
        }
        
        list.models <- list()
        for(f in 1:nfolds) {
          #each fold
          if(verbose == TRUE) {
            cat(paste('xgcv: fold:', f,'\n'))
          }
          train.data.int <- int.crossval.data[[1]][[f]]
          #test.data.int <- crossval.data[[2]][[f]]
          train.labels.int <- int.crossval.data[[3]][[f]]
          #test.labels.int<- crossval.data[[4]][[f]]
          
          model <- xgboost(data=as.matrix(train.data.int), label=as.numeric(train.labels.int), verbose=0, nrounds=max(iters),
                           eta=et, max.depth=tr, objective='binary:logistic', nthread=64)
          list.models[[f]] <- model
          
        } 
        
        int.auc <- -Inf
        int.it <- max(iters)
        int.aucs <- vector()
        for(it in iters) {
          aucs <- vector()
          for(f in 1:nfolds) {
            test.data.int <- int.crossval.data[[2]][[f]]
            test.labels.int<- int.crossval.data[[4]][[f]]
            resp <- predict(list.models[[f]], as.matrix(test.data.int), ntreelimit = it)
            roc.model <- roc(as.numeric(test.labels.int), as.numeric(resp))
            aucs[f] <- roc.model$auc
          }
          
          if(mean(aucs) > int.auc) {
            int.auc <- mean(aucs)
            int.it <- it
            int.aucs <- aucs
          }
          
        }
        
        xgb.auc <- int.aucs
        int.iters <- int.it
        
        #Return highest mean C-stat
          if(verbose == TRUE) {
            cat(paste('xgcv: prev.auc:', best.auc,' this round:', mean(xgb.auc),' iters:', int.iters, '\n'))
            cat(paste('-----------------------\n'))
          }
          
          if(mean(xgb.auc) >= best.auc) {
            best.auc <- mean(xgb.auc)
	          best.tr <- tr
            best.eta <- et
            best.iter <- int.iters#it
          }
          
      #}
    }
  }
  
  #Return parameters in a vector
  list.results <- list()
  list.results[[1]] <- best.eta
  list.results[[2]] <- best.tr
  list.results[[3]] <- best.iter
  return(list.results)
  
}

xgcv_par <- function(train.data, train.labels, tree.depth, eta, iters, verbose, nfolds = 5, ncores=8, nthread=32, linux=FALSE) {
 
  
  list.options <- list()
  i <- 1
  for(dp in tree.depth) {
    for(et in eta) {
      
        values <- c(dp, et)
        list.options[[i]] <- values
        i <- i+1
    }
  }
  int.crossval.data <- crossval.list(train.data, train.labels, nfolds)
  
  if(!linux) {
    registerDoParallel(cores=ncores)
  } else {
    c1 <- makePSOCKcluster(ncores)
    registerDoParallel(c1)
  }
  best.auc <- -Inf
  best.tr <- NA
  best.eta <- NA
  best.iter <- NA
  
  all.results <- foreach(opt=list.options, .combine=rbind) %dopar% {
    library(xgboost)
    library(pROC)
    
    list.models <- list()
    for(f in 1:nfolds) {
      #each fold
      #if(verbose == TRUE) {
      #  cat(paste('xgcv: fold:', f,'\n'))
      #}
      train.data.int <- int.crossval.data[[1]][[f]]
      #test.data.int <- crossval.data[[2]][[f]]
      train.labels.int <- int.crossval.data[[3]][[f]]
      #test.labels.int<- crossval.data[[4]][[f]]
      
      model <- xgboost(data=as.matrix(train.data.int), label=as.numeric(train.labels.int), verbose=0, nrounds=max(iters),
                       eta=opt[2], max.depth=opt[1], objective='binary:logistic', nthread=nthread,
                       save_period = NULL, save_name = NULL)
      list.models[[f]] <- model
      
    } 
    
    int.auc <- -Inf
    int.it <- max(iters)
    int.aucs <- vector()
    for(it in iters) {
      aucs <- vector()
      for(f in 1:nfolds) {
        test.data.int <- int.crossval.data[[2]][[f]]
        test.labels.int<- int.crossval.data[[4]][[f]]
        resp <- predict(list.models[[f]], as.matrix(test.data.int), ntreelimit = it)
        roc.model <- roc(as.numeric(test.labels.int), as.numeric(resp))
        aucs[f] <- roc.model$auc
      }
      
      if(mean(aucs) > int.auc) {
        int.auc <- mean(aucs)
        int.it <- it
        int.aucs <- aucs
      }
      
    }
    
    opt.vals <- vector()
    opt.vals <- c(int.auc, int.it)
    
  } #End of foreach
  
  
  df <- data.frame(all.results)
  idx.sds <- which(df[,1] > max(df[,1]) - sd(df[,1]))
  #val.idx <- which(df[idx.sds,2] == min(df[idx.sds,2]))
  #val <- df[val.idx,1]
  val <- max(df[,1])
  val.idx <- which(val == df[,1])
  final.opts <- list.options[[val.idx]]
  best.eta <- final.opts[2]
  best.tr <- final.opts[1]
  best.it <- df[val.idx,2]#final.opts[3]
  return(c(best.eta, best.tr, best.it))
  
}
