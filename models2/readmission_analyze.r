MODE <- '72hrs'
# save('list.predglm',
#      'list.predxgb',
#      'list.rocglm',
#      'list.rocxgb',
#      'list.xgbparam',
#      'list.glm',
#      'list.xgb',
#      'list.featurenames',
#      'list.labels',
#      
#      'list.train.in',
#      'list.test.in',
#      'list.train.out',
#      'list.test.out',
#      
#      file=paste('./results/aucs_FEB2018_', MODE,'_data.RData', sep=''))
source('fMeasure.R')
library(pROC)
library(glmnet)
library(xgboost)
source('calcDeciles.R')

load(paste('./results/aucs_FEB2018_', MODE,'_data.RData', sep=''))

list.f.glm <- list()
list.f.xgb <- list()
aucs.glm <- vector()
aucs.xgb <- vector()
for(f in 1:5) {
  list.f.glm[[f]] <- allROC_par(list.predglm[[f]], list.labels[[f]][[2]])[[1]]
  list.f.xgb[[f]] <- allROC_par(list.predxgb[[f]], list.labels[[f]][[2]])[[1]]
  aucs.glm[f] <- list.rocglm[[f]]$auc
  aucs.xgb[f] <- list.rocxgb[[f]]$auc
}

precision.xgb <- vector()
precision.glm <- vector()
recall.xgb <- vector()
recall.glm <- vector()

f.xgb <- vector()
f.glm <- vector()
for(f in 1:5) {
  f.xgb[f] <- list.f.xgb[[f]]$f.score[1]
  f.glm[f] <- list.f.glm[[f]]$f.score[1]
  
  precision.xgb[f] <- list.f.xgb[[f]]$pr[1]
  precision.glm[f] <- list.f.glm[[f]]$pr[1]
  
  recall.xgb[f] <- list.f.xgb[[f]]$re[1]
  recall.glm[f] <- list.f.glm[[f]]$re[1]
  
}

library(SpecsVerification)
#list.labels
#list.predglm
#list.predxgb

rel <- vector()
res <- vector()
unc <- vector()
score <- vector()
labels <- vector()
probs.xgb <- vector()

for(f in 1:5) {
  labels <- c(labels, list.labels[[f]][[2]])
  probs.xgb <- c(probs.xgb, list.predxgb[[f]])
  bd <- BrierDecomp(list.predxgb[[f]], list.labels[[f]][[2]])
  rel[f] <- bd[1,1]
  res[f] <- bd[1,2]
  unc[f] <- bd[1,3]
  score[f] <- rel[f] - res[f] + unc[f]
  
}

bd.total <- BrierDecomp(probs.xgb, labels)

rel <- c(rel, bd.total[1,1])
res <- c(res, bd.total[1,2])
unc <- c(unc, bd.total[1,3])
score <- c(score, bd.total[1,1] - bd.total[1,2] + bd.total[1,3])

bd.df <- data.frame(rel, res, unc, score)
rownames(bd.df) <- c('fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'total')

source('calcDeciles.R')
for(f in 1:5) {
  df <- calcDeciles(list.predxgb[[f]], list.labels[[f]][[2]]) [[1]]
  write.csv(df, file=paste('./results/calibration_fold_', f,'_', MODE, '.csv', sep=''))
}

