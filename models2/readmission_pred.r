#MODE <- '24hrs'
library(doParallel)
library(foreach)
registerDoParallel(cores=32)
#list.mode.results <- foreach(MODE=c('24hrs', '48hrs', '72hrs', '24hrs~72hrs', '7days', '30days')) %dopar% {
list.mode.results <- foreach(MODE=c('48hrs', '7days', 'Bounceback')) %dopar% {
#for(MODE in c('24hrs', '48hrs', '72hrs', '24hrs~72hrs', '7days', '30days', 'bounceback')) {
Prefix <- './FOLDS_PROBS/IsReadmitted_'

library(glmnet)
library(xgboost)
library(pROC)
source('./xgboostWrapper.R')
source('./catToBin.R')

list.predglm <- list()
list.predxgb <- list()
list.rocglm <- list()
list.rocxgb <- list()
list.xgbparam <- list()
list.glm <- list()
list.xgb <- list()
list.featurenames <- list()
list.labels <- list()

list.train.in <- list()
list.test.in <- list()
list.train.out <- list()
list.test.out <- list()

for(f in 1:5) {
  cat(paste('fold:', f, '\n'))
  train.data <- read.csv(paste(Prefix, MODE, '/fold_', f, '/training.csv', sep=''))
  test.data <- read.csv(paste(Prefix, MODE, '/fold_', f, '/testing.csv', sep=''))
  
  cols.remove <- which(grepl('HADM_ID', colnames(train.data)) |
                         grepl('ICUSTAY_ID', colnames(train.data)) |
                         grepl('INTIME', colnames(train.data)) |
                         grepl('OUTTIME', colnames(train.data)) |
                         grepl('SUBJECT_ID', colnames(train.data)) |
                         grepl('INSURANCE', colnames(train.data)) |
                         grepl('RELIGION', colnames(train.data)) |
                         grepl('LANGUAGE', colnames(train.data)) |
                         grepl('MARITAL_STATUS', colnames(train.data)))
  
  train.data <- train.data[,-cols.remove]
  test.data <- test.data[,-cols.remove]
  cols.out <- which(grepl('IsReadmitted_', colnames(train.data)))
  
  train.out <- train.data[,cols.out]
  test.out <- test.data[,cols.out]
  
  train.in <- train.data[,-cols.out]
  test.in <- test.data[,-cols.out]
  
  first.care <- catToBin(train.in$FIRST_CAREUNIT)
  colnames(first.care) <- paste('FIRST_CARE_', colnames(first.care), sep='')
  first.care.test <- catToBin(test.in$FIRST_CAREUNIT)
  colnames(first.care.test) <- paste('FIRST_CARE_', colnames(first.care.test), sep='')  
  
  train.in <- cbind(train.in, first.care)
  test.in <- cbind(test.in, first.care.test)  
  
  train.in$FIRST_CAREUNIT <- NULL
  test.in$FIRST_CAREUNIT <- NULL
  
  adm.type <- catToBin(train.in$ADMISSION_TYPE)
  colnames(adm.type) <- paste('ADMISSION_TYPE_', colnames(adm.type), sep='')
  adm.type.test <- catToBin(test.in$ADMISSION_TYPE)
  colnames(adm.type.test) <- paste('ADMISSION_TYPE', colnames(adm.type.test), sep='')  
  
  train.in <- cbind(train.in, adm.type)
  test.in <- cbind(test.in, adm.type.test)  
  
  train.in$ADMISSION_TYPE <- NULL
  test.in$ADMISSION_TYPE <- NULL
  
  ethn <- catToBin(train.in$ETHNICITY)
  colnames(ethn) <- paste('ETHNICITY_', colnames(ethn), sep='')
  ethn.test <- catToBin(test.in$ETHNICITY)
  colnames(ethn.test) <- paste('ETHNICITY_', colnames(ethn.test), sep='')  
  
  #Rebuild Ethnicity into groups
  # ethn.nativeamerican <- rowSums(ethn[,c(1,2)])
  # ethn.asian <- rowSums(ethn[,c(3,4,5,6,7,8,9,10,11,12)])
  # ethn.black <- rowSums(ethn[,c(13,14,15,16,17)])
  # ethn.hispanic <- rowSums(ethn[,seq(18,27,1)])
  # ethn.white <- rowSums(ethn[,c(33,34,37,38,39,40,41)])
  # ethn.other <- rowSums(ethn[,c(28,29,30,31,32,35,36)])
  # 
  # ethn.nativeamerican[which(ethn.nativeamerican > 1)] <- 1
  # ethn.asian[which(ethn.asian > 1)] <- 1
  # ethn.black[which(ethn.black > 1)] <- 1
  # ethn.hispanic[which(ethn.hispanic > 1)] <- 1
  # ethn.white[which(ethn.white > 1)] <- 1
  # ethn.other[which(ethn.other > 1)] <- 1
  # 
  # ethn.nativeamerican.test <- rowSums(ethn.test[,c(1,2)])
  # ethn.asian.test<- rowSums(ethn.test[,c(3,4,5,6,7,8,9,10,11,12)])
  # ethn.black.test <- rowSums(ethn.test[,c(13,14,15,16,17)])
  # ethn.hispanic.test <- rowSums(ethn.test[,seq(18,27,1)])
  # ethn.white.test <- rowSums(ethn.test[,c(33,34,37,38,39,40,41)])
  # ethn.other.test <- rowSums(ethn.test[,c(28,29,30,31,32,35,36)])
  # 
  # ethn.nativeamerican.test[which(ethn.nativeamerican.test > 1)] <- 1
  # ethn.asian.test[which(ethn.asian.test > 1)] <- 1
  # ethn.black.test[which(ethn.black.test > 1)] <- 1
  # ethn.hispanic.test[which(ethn.hispanic.test > 1)] <- 1
  # ethn.white.test[which(ethn.white.test > 1)] <- 1
  # ethn.other.test[which(ethn.other.test > 1)] <- 1
  # 
  # train.in <- cbind(train.in, ethn.nativeamerican, ethn.asian, ethn.black, ethn.hispanic, ethn.white, ethn.other)
  # test.in <- cbind(test.in, ethn.nativeamerican.test, ethn.asian.test, ethn.black.test, ethn.hispanic.test, ethn.white.test, ethn.other.test)
  # 
  # 
  train.in$ETHNICITY <- NULL
  test.in$ETHNICITY <- NULL
  
  train.in <- data.matrix(train.in)
  test.in <- data.matrix(test.in)
  
  col.means <- colMeans(train.in ,na.rm=TRUE)
  for(j in 1:dim(train.in)[2]) {
    if(any(is.na(train.in[,j]))) {
      train.in[,j][which(is.na(train.in[,j]))] <- col.means[j]
    }
  }
  
  for(j in 1:dim(test.in)[2]) {
    if(any(is.na(test.in[,j]))) {
      test.in[,j][which(is.na(test.in[,j]))] <- col.means[j]
    }
  }
  
  if(MODE == '24hrs') {
    train.labels <- train.out$IsReadmitted_24hrs
    test.labels <- test.out$IsReadmitted_24hrs
  } else if(MODE== '48hrs') {
    train.labels <- train.out$IsReadmitted_48hrs
    test.labels <- test.out$IsReadmitted_48hrs
  } else if(MODE == '72hrs') {
    train.labels <- train.out$IsReadmitted_72hrs
    test.labels <- test.out$IsReadmitted_72hrs
  } else if(MODE == '24hrs~72hrs') {
    train.labels <- train.out$IsReadmitted_24hrs.72hrs
    test.labels <- test.out$IsReadmitted_24hrs.72hrs
  } else if(MODE == '7days') {
    train.labels <- train.out$IsReadmitted_7days
    test.labels <- test.out$IsReadmitted_7days
  } else if(MODE == '30days') {
    train.labels <- train.out$IsReadmitted_30days
    test.labels <- test.out$IsReadmitted_30days
  } else {
    train.labels <- train.out$IsReadmitted_Bounceback
    test.labels <- test.out$IsReadmitted_Bounceback
  }
  
  
  w.samples <- rep(1, dim(train.in)[1])
  w.samples[which(train.labels == 1)] <- round((length(train.labels) - sum(train.labels))/sum(train.labels)) 
    #round((length(train.labels)-sum(train.labels))/sum(train.labels))
  library(doParallel)
  library(foreach)
  registerDoParallel(cores=32)
  source('fMeasure.R')
  cat(paste('fold:', f, ' GLM\n'))
  glm.model <- cv.glmnet(x=train.in,y=as.factor(train.labels), family='binomial',
                         type.measure='auc', parallel=TRUE, weights=w.samples)
  glm.pred <- predict(glm.model, test.in, type='response')
  glm.roc <- roc(as.numeric(test.labels), as.numeric(glm.pred))
  glm.f <- allROC_par(as.numeric(glm.pred), as.numeric(test.labels))[[1]]
  cat(paste('fold:', f, ' GLM ROC:', glm.roc$auc, '\n'))
 
  xgb.params <- xgcv_par(train.in, train.labels, seq(1,6,1), c(0.1), c(50,100,150, 200), FALSE, nfolds = 5, ncores=16, nthread=32, linux=TRUE)
    
  xgb.model <- xgboost(train.in, as.numeric(train.labels), verbose=0, nrounds=xgb.params[3], 
                       eta=0.1, max.depth=xgb.params[2],  objective='binary:logistic',nthread=32,
                       save_period=NULL, save_name=NULL)
  xgb.pred <- predict(xgb.model, test.in)
  xgb.roc <- roc(as.numeric(test.labels), as.numeric(xgb.pred))
  xgb.f <- allROC_par(as.numeric(xgb.pred), as.numeric(test.labels))[[1]]
  cat(paste('fold:', f, ' XGB:', xgb.roc$auc, '\n'))
  
  list.featurenames[[f]] <- colnames(train.in)
  list.predglm[[f]] <- glm.pred
  list.predxgb[[f]] <- xgb.pred
  list.rocglm[[f]] <- glm.roc
  list.rocxgb[[f]] <- xgb.roc
  list.xgbparam[[f]] <- xgb.params
  list.glm[[f]] <- glm.model
  list.xgb[[f]] <- xgb.model
  list.labels[[f]] <- list(train.labels, test.labels)
  
  list.train.in[[f]] <- train.in
  list.test.in[[f]] <- test.in
  list.train.out[[f]] <- train.out
  list.test.out[[f]] <- test.out
}

save('list.predglm',
     'list.predxgb',
     'list.rocglm',
     'list.rocxgb',
     'list.xgbparam',
     'list.glm',
     'list.xgb',
     'list.featurenames',
     'list.labels',
     
     'list.train.in',
     'list.test.in',
     'list.train.out',
     'list.test.out',

     file=paste('./results/aucs_FEB2018_', MODE,'_data.RData', sep=''))

}