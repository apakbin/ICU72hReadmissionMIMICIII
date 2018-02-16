list.fullrank <- list()
MODE <- '30days'
load(paste('./results/aucs_FEB2018_', MODE,'_data.RData', sep=''))
library(xgboost)
for(f in 1:5) {
  list.fullrank[[f]] <- xgb.importance(feature_names=list.featurenames[[f]], model=list.xgb[[f]])
}

vars <- list.featurenames[[1]]
rankmat <- matrix(NA, nrow=length(vars), ncol=5)

for(v in 1:length(vars)) {
  rankvec <- vector()
  for(f in 1:5) {
    temp <- which(vars[v] == list.fullrank[[f]]$Feature)
    if(length(temp) == 0) {
      rankvec[f] <- NA
    } else {
      rankvec[f] <- temp[1]
    }
    
  }
  rankmat[v,] <- rankvec
}

rankdf <- data.frame(rankmat)
fullmean <- rowMeans(rankdf, na.rm=FALSE)
NAMean <- rowMeans(rankdf, na.rm=TRUE)
fullsd <- apply(rankdf, 1, sd, na.rm=FALSE)
NAsd <- apply(rankdf, 1, sd, na.rm=TRUE)

rankdf$fullmean <- fullmean
rankdf$fullsd <- fullsd
rankdf$NAMean <- NAMean
rankdf$NAsd <- NAsd

colnames(rankdf) <- c('fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'Fullmean', 'Fullstd', 'NAMean', 'NAstd')
rownames(rankdf) <- vars
write.csv(rankdf, file=paste('./results/variables_MODE_', MODE, '_6FEB2018.csv',sep=''))
