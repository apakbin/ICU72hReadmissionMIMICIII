#This calculates the f-measure
library(foreach)
library(doParallel)
fscore <- function(tp, tn, fp, fn) {
  if(tp + fn == 0) {
    re <- 0
  } else {
    re <- (tp / (tp + fn))    
  }
  if(tp + fp == 0) {
    pr <- 0
  } else {
    pr <- (tp / (tp + fp))  
  }
  
  if(tn + fp == 0){
    spec <- 0
  } else {
    spec <- (tn / (tn + fp))
  }
  
  if(pr + re == 0)
  {
    f <- 0
    res <- data.frame(f, pr, re, spec)
    return(res)
  } else {
    f <- (2 * (pr * re) / (pr + re) )
    res <- data.frame(f, pr, re, spec)
    return( res )
  }
}

ratesOfPredict <- function(predicted, gt) {
  tp <- 0
  fp <- 0
  tn <- 0
  fn <- 0
  
  tp <- length(which((predicted == gt) & (predicted == 1)))
  tn <- length(which((predicted == gt) & (predicted == 0)))
  fp <- length(which((predicted != gt) & (predicted == 1)))
  fn <- length(which((predicted != gt) & (predicted == 0)))
  
  return(data.frame(tp, tn, fp, fn))
}

allROC_par <- function(response, gt) {

  registerDoParallel(cores=8)
  
  min.val <- min(response)
  max.val <- max(response)
  vals <- seq(min.val, max.val, (max.val - min.val)/100)
  max.f <- 0
  best.res <- NA
  best.val <- min.val
  all.vals <- foreach(i=1:length(vals), .combine=rbind) %dopar% {
    v <- vals[i]
    pred <- rep(0, length(response))
    pred[which(response >= v)] <- 1
    #values <- ratesOfPredict(pred, gt)
    tp <- 0
    fp <- 0
    tn <- 0
    fn <- 0
    
    tp <- length(which((pred == gt) & (pred == 1)))
    tn <- length(which((pred == gt) & (pred == 0)))
    fp <- length(which((pred != gt) & (pred == 1)))
    fn <- length(which((pred != gt) & (pred == 0)))
    
    if(tp + fn == 0) {
      re <- 0
    } else {
      re <- (tp / (tp + fn))    
    }
    if(tp + fp == 0) {
      pr <- 0
    } else {
      pr <- (tp / (tp + fp))  
    }
    
    if(tn + fp == 0){
      spec <- 0
    } else {
      spec <- (tn / (tn + fp))
    }
    
    if(pr + re == 0)
    {
      f.score <- 0
      #res <- data.frame(f, pr, re, spec)
      #return(res)
    } else {
      f.score <- (2 * (pr * re) / (pr + re) )
      #res <- data.frame(f, pr, re, spec)
      #return( res )
    }
    
    #results <- fscore(values$tp, values$tn, values$fp, values$fn)
    data.frame(min.val, max.val, v, tp, tn, fp, fn, pr, re, spec, f.score)
  }
  
  best.res <- all.vals[which(all.vals$f.score == max(all.vals$f.score)),]
  
  res.final <- list()
  res.final[[1]] <- best.res
  res.final[[2]] <- all.vals
  return(res.final)
}

generateLabels <- function(response, thresh) {
  
}
