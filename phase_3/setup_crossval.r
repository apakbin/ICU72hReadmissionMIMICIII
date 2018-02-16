#Create Cross Validation Folds
crossval.list <- function(dataset, labels, nfolds) {
  #Creates a stratified k-fold cross-validation
  positives <- which(labels == 1)
  negatives <- which(labels == 0)
  
  data.positives <- dataset[positives,]
  data.negatives <- dataset[negatives,]
  labels.positives <- labels[positives]
  labels.negatives <- labels[negatives]
  
  split <- 1/nfolds
  numpos <- round(split*length(positives))
  numneg <- round(split*length(negatives))
  
  data.pos <- list()
  data.neg <- list()
  labels.pos <- list()
  labels.neg <- list()
  idx.pos <- list()
  idx.neg <- list()
  
  for (i in 1:(nfolds-1)) {

    cv.fold.idx.pos <- sort(sample(dim(data.positives)[1], numpos, replace=FALSE))
    cv.fold.idx.neg <- sort(sample(dim(data.negatives)[1], numneg, replace=FALSE))
    
    fold.pos <- data.positives[cv.fold.idx.pos,]
    fold.neg <- data.negatives[cv.fold.idx.neg,]
    fold.labels.pos <- labels.positives[cv.fold.idx.pos]
    fold.labels.neg <- labels.negatives[cv.fold.idx.neg]
    
    data.positives <- data.positives[-cv.fold.idx.pos,]
    data.negatives <- data.negatives[-cv.fold.idx.neg,]
    labels.positives <- labels.positives[-cv.fold.idx.pos]
    labels.negatives <- labels.negatives[-cv.fold.idx.neg]
    
    data.pos[[i]] <- fold.pos
    data.neg[[i]] <- fold.neg
    idx.pos[[i]] <- cv.fold.idx.pos
    idx.neg[[i]] <- cv.fold.idx.neg
    labels.pos[[i]] <- fold.labels.pos
    labels.neg[[i]] <- fold.labels.neg
  }
  i <- i + 1
  
  data.pos[[i]] <- data.positives
  data.neg[[i]] <- data.negatives
  labels.pos[[i]] <- labels.positives
  labels.neg[[i]] <- labels.negatives
  
  #Data split, now create fold data
  train.data.folds <- list()
  test.data.folds <- list()
  train.labels.folds <- list()
  test.labels.folds <- list()
  
  for(i in 1:nfolds) {
    
    test.data <- rbind(data.pos[[i]], data.neg[[i]])
    test.labels <- c(labels.pos[[i]], labels.neg[[i]])
    
    train.data <- vector()
    train.labels <- vector()
    for(j in 1:nfolds) {
      if(j != i) {
        train.data <- rbind(train.data,
                            data.pos[[j]], data.neg[[j]])
        train.labels <- c(train.labels, labels.pos[[j]], labels.neg[[j]])
      }
    }
    
    train.data.folds[[i]] <- train.data
    train.labels.folds[[i]] <- train.labels
    
    test.data.folds[[i]] <- test.data
    test.labels.folds[[i]] <- test.labels
  }
  
  folds.lists <- list()
  folds.lists[[1]] <- train.data.folds
  folds.lists[[2]] <- test.data.folds
  folds.lists[[3]] <- train.labels.folds
  folds.lists[[4]] <- test.labels.folds
  folds.lists[[5]] <- idx.pos
  folds.lists[[6]] <- idx.neg
  folds.lists[[7]] <- positives
  folds.lists[[8]] <- negatives

  return(folds.lists)
}
