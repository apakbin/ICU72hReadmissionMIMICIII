calcDeciles <- function(probabilities, labels) {
  probs.sorted <- probabilities[order(probabilities)]
  labels.sorted <- labels[order(probabilities)]
  
  deciles <- quantile(probs.sorted, probs=seq(0,1,.1))
  
  dec1.idx <- which(probs.sorted < deciles[2])
  dec2.idx <- which((probs.sorted < deciles[3]) & (probs.sorted >= deciles[2]))
  dec3.idx <- which((probs.sorted < deciles[4]) & (probs.sorted >= deciles[3]))
  dec4.idx <- which((probs.sorted < deciles[5]) & (probs.sorted >= deciles[4]))
  dec5.idx <- which((probs.sorted < deciles[6]) & (probs.sorted >= deciles[5]))
  dec6.idx <- which((probs.sorted < deciles[7]) & (probs.sorted >= deciles[6]))
  dec7.idx <- which((probs.sorted < deciles[8]) & (probs.sorted >= deciles[7]))
  dec8.idx <- which((probs.sorted < deciles[9]) & (probs.sorted >= deciles[8]))
  dec9.idx <- which((probs.sorted < deciles[10]) & (probs.sorted >= deciles[9]))
  dec10.idx <- which(probs.sorted >= deciles[10])
  
  #response
  dec1.resp <- probs.sorted[dec1.idx]
  dec2.resp <- probs.sorted[dec2.idx]
  dec3.resp <- probs.sorted[dec3.idx]
  dec4.resp <- probs.sorted[dec4.idx]
  dec5.resp <- probs.sorted[dec5.idx]
  dec6.resp <- probs.sorted[dec6.idx]
  dec7.resp <- probs.sorted[dec7.idx]
  dec8.resp <- probs.sorted[dec8.idx]
  dec9.resp <- probs.sorted[dec9.idx]
  dec10.resp <- probs.sorted[dec10.idx]
 
  dec1.labs <- labels.sorted[dec1.idx]
  dec2.labs <- labels.sorted[dec2.idx]
  dec3.labs <- labels.sorted[dec3.idx]
  dec4.labs <- labels.sorted[dec4.idx]
  dec5.labs <- labels.sorted[dec5.idx]
  dec6.labs <- labels.sorted[dec6.idx]
  dec7.labs <- labels.sorted[dec7.idx]
  dec8.labs <- labels.sorted[dec8.idx]
  dec9.labs <- labels.sorted[dec9.idx]
  dec10.labs <- labels.sorted[dec10.idx]
  
  predicted <- vector()
  observed <- vector()
  
  predicted <- 100* c(mean(dec1.resp),mean(dec2.resp),mean(dec3.resp),mean(dec4.resp),mean(dec5.resp),
                 mean(dec6.resp),mean(dec7.resp),mean(dec8.resp),mean(dec9.resp),mean(dec10.resp))
  
  dec1.rates <- 100* ((sum(dec1.labs))/(length(dec1.labs)))
  dec2.rates <- 100* ((sum(dec2.labs))/(length(dec2.labs)))
  dec3.rates <- 100* ((sum(dec3.labs))/(length(dec3.labs)))
  dec4.rates <- 100* ((sum(dec4.labs))/(length(dec4.labs)))
  dec5.rates <- 100* ((sum(dec5.labs))/(length(dec5.labs)))
  dec6.rates <- 100* ((sum(dec6.labs))/(length(dec6.labs)))
  dec7.rates <- 100* ((sum(dec7.labs))/(length(dec7.labs)))
  dec8.rates <- 100* ((sum(dec8.labs))/(length(dec8.labs)))
  dec9.rates <- 100* ((sum(dec9.labs))/(length(dec9.labs)))
  dec10.rates <- 100* ((sum(dec10.labs))/(length(dec10.labs)))
  
  observed <- c(dec1.rates,dec2.rates,dec3.rates,dec4.rates,dec5.rates,dec6.rates,dec7.rates,dec8.rates,dec9.rates,dec10.rates)
  
  calibration <- data.frame(predicted, observed)
  colnames(calibration) <- c('predicted rates', 'observed rates')
  rownames(calibration) <- c('decile 1', 'decile 2', 'decile 3', 'decile 4', 'decile 5', 
                             'decile 6', 'decile 7', 'decile 8', 'decile 9', 'decile 10')
  list.res <- list()
  list.res[[1]] <- calibration
  list.res[[2]] <- probs.sorted
  list.res[[3]] <- labels.sorted
  return(list.res)
}

sortDeciles <- function(probabilities, labels, deciles) {
  probs.sorted <- probabilities[order(probabilities)]
  labels.sorted <- labels[order(probabilities)]
  
  #deciles <- quantile(probs.sorted, probs=seq(0,1,.1))
  
  dec1.idx <- which(probs.sorted < deciles[2])
  dec2.idx <- which((probs.sorted < deciles[3]) & (probs.sorted >= deciles[2]))
  dec3.idx <- which((probs.sorted < deciles[4]) & (probs.sorted >= deciles[3]))
  dec4.idx <- which((probs.sorted < deciles[5]) & (probs.sorted >= deciles[4]))
  dec5.idx <- which((probs.sorted < deciles[6]) & (probs.sorted >= deciles[5]))
  dec6.idx <- which((probs.sorted < deciles[7]) & (probs.sorted >= deciles[6]))
  dec7.idx <- which((probs.sorted < deciles[8]) & (probs.sorted >= deciles[7]))
  dec8.idx <- which((probs.sorted < deciles[9]) & (probs.sorted >= deciles[8]))
  dec9.idx <- which((probs.sorted < deciles[10]) & (probs.sorted >= deciles[9]))
  dec10.idx <- which(probs.sorted >= deciles[10])
  
  #response
  dec1.resp <- probs.sorted[dec1.idx]
  dec2.resp <- probs.sorted[dec2.idx]
  dec3.resp <- probs.sorted[dec3.idx]
  dec4.resp <- probs.sorted[dec4.idx]
  dec5.resp <- probs.sorted[dec5.idx]
  dec6.resp <- probs.sorted[dec6.idx]
  dec7.resp <- probs.sorted[dec7.idx]
  dec8.resp <- probs.sorted[dec8.idx]
  dec9.resp <- probs.sorted[dec9.idx]
  dec10.resp <- probs.sorted[dec10.idx]
  
  dec1.labs <- labels.sorted[dec1.idx]
  dec2.labs <- labels.sorted[dec2.idx]
  dec3.labs <- labels.sorted[dec3.idx]
  dec4.labs <- labels.sorted[dec4.idx]
  dec5.labs <- labels.sorted[dec5.idx]
  dec6.labs <- labels.sorted[dec6.idx]
  dec7.labs <- labels.sorted[dec7.idx]
  dec8.labs <- labels.sorted[dec8.idx]
  dec9.labs <- labels.sorted[dec9.idx]
  dec10.labs <- labels.sorted[dec10.idx]
  
  predicted <- vector()
  observed <- vector()
  std <- vector()
  lower <- vector()
  upper <- vector()
  
  predicted <- 100* c(mean(dec1.resp),mean(dec2.resp),mean(dec3.resp),mean(dec4.resp),mean(dec5.resp),
                      mean(dec6.resp),mean(dec7.resp),mean(dec8.resp),mean(dec9.resp),mean(dec10.resp))
  std <- 100* c(sd(dec1.resp),sd(dec2.resp),sd(dec3.resp),sd(dec4.resp),sd(dec5.resp),
                      sd(dec6.resp),sd(dec7.resp),sd(dec8.resp),sd(dec9.resp),sd(dec10.resp))
  lower <- 100* c(t.test(dec1.resp)$conf.int[1],t.test(dec2.resp)$conf.int[1],t.test(dec3.resp)$conf.int[1],t.test(dec4.resp)$conf.int[1],t.test(dec5.resp)$conf.int[1],
                t.test(dec6.resp)$conf.int[1],t.test(dec7.resp)$conf.int[1],t.test(dec8.resp)$conf.int[1],t.test(dec9.resp)$conf.int[1],t.test(dec10.resp)$conf.int[1])
  upper <- 100* c(t.test(dec1.resp)$conf.int[2],t.test(dec2.resp)$conf.int[2],t.test(dec3.resp)$conf.int[2],t.test(dec4.resp)$conf.int[2],t.test(dec5.resp)$conf.int[2],
                t.test(dec6.resp)$conf.int[2],t.test(dec7.resp)$conf.int[2],t.test(dec8.resp)$conf.int[2],t.test(dec9.resp)$conf.int[2],t.test(dec10.resp)$conf.int[2])
  dec1.rates <- 100* ((sum(dec1.labs))/(length(dec1.labs)))
  dec2.rates <- 100* ((sum(dec2.labs))/(length(dec2.labs)))
  dec3.rates <- 100* ((sum(dec3.labs))/(length(dec3.labs)))
  dec4.rates <- 100* ((sum(dec4.labs))/(length(dec4.labs)))
  dec5.rates <- 100* ((sum(dec5.labs))/(length(dec5.labs)))
  dec6.rates <- 100* ((sum(dec6.labs))/(length(dec6.labs)))
  dec7.rates <- 100* ((sum(dec7.labs))/(length(dec7.labs)))
  dec8.rates <- 100* ((sum(dec8.labs))/(length(dec8.labs)))
  dec9.rates <- 100* ((sum(dec9.labs))/(length(dec9.labs)))
  dec10.rates <- 100* ((sum(dec10.labs))/(length(dec10.labs)))
  
  observed <- c(dec1.rates,dec2.rates,dec3.rates,dec4.rates,dec5.rates,dec6.rates,dec7.rates,dec8.rates,dec9.rates,dec10.rates)
  
  calibration <- data.frame(predicted, std, lower, upper, observed)
  colnames(calibration) <- c('predicted rates', 'standard deviation', 'lower', 'upper', 'observed rates')
  rownames(calibration) <- c('decile 1', 'decile 2', 'decile 3', 'decile 4', 'decile 5', 
                             'decile 6', 'decile 7', 'decile 8', 'decile 9', 'decile 10')
  list.res <- list()
  list.res[[1]] <- calibration
  list.res[[2]] <- probs.sorted
  list.res[[3]] <- labels.sorted
  return(list.res)
}

calcQuartiles <- function(probabilities, labels) {
  probs.sorted <- probabilities[order(probabilities)]
  labels.sorted <- labels[order(probabilities)]
  
  deciles <- quantile(probs.sorted, probs=seq(0,1,.25))
  
  dec1.idx <- which(probs.sorted < deciles[2])
  dec2.idx <- which((probs.sorted < deciles[3]) & (probs.sorted >= deciles[2]))
  dec3.idx <- which((probs.sorted < deciles[4]) & (probs.sorted >= deciles[3]))
  dec4.idx <- which(probs.sorted >= deciles[4])
  
  
  #response
  dec1.resp <- probs.sorted[dec1.idx]
  dec2.resp <- probs.sorted[dec2.idx]
  dec3.resp <- probs.sorted[dec3.idx]
  dec4.resp <- probs.sorted[dec4.idx]
  
  
  dec1.labs <- labels.sorted[dec1.idx]
  dec2.labs <- labels.sorted[dec2.idx]
  dec3.labs <- labels.sorted[dec3.idx]
  dec4.labs <- labels.sorted[dec4.idx]
  
  
  predicted <- vector()
  observed <- vector()
  
  predicted <- 100* c(mean(dec1.resp),mean(dec2.resp),mean(dec3.resp),mean(dec4.resp))
  
  dec1.rates <- 100* ((sum(dec1.labs))/(length(dec1.labs)))
  dec2.rates <- 100* ((sum(dec2.labs))/(length(dec2.labs)))
  dec3.rates <- 100* ((sum(dec3.labs))/(length(dec3.labs)))
  dec4.rates <- 100* ((sum(dec4.labs))/(length(dec4.labs)))
  
  observed <- c(dec1.rates,dec2.rates,dec3.rates,dec4.rates)
  
  calibration <- data.frame(predicted, observed)
  colnames(calibration) <- c('predicted rates', 'observed rates')
  rownames(calibration) <- c('decile 1', 'decile 2', 'decile 3', 'decile 4')
  list.res <- list()
  list.res[[1]] <- calibration
  list.res[[2]] <- probs.sorted
  list.res[[3]] <- labels.sorted
  return(list.res)
}

sortQuartiles <- function(probabilities, labels, deciles) {
  probs.sorted <- probabilities[order(probabilities)]
  labels.sorted <- labels[order(probabilities)]
  
  #deciles <- quantile(probs.sorted, probs=seq(0,1,.25))
  
  dec1.idx <- which(probs.sorted < deciles[2])
  dec2.idx <- which((probs.sorted < deciles[3]) & (probs.sorted >= deciles[2]))
  dec3.idx <- which((probs.sorted < deciles[4]) & (probs.sorted >= deciles[3]))
  dec4.idx <- which(probs.sorted >= deciles[4])
  
  
  #response
  dec1.resp <- probs.sorted[dec1.idx]
  dec2.resp <- probs.sorted[dec2.idx]
  dec3.resp <- probs.sorted[dec3.idx]
  dec4.resp <- probs.sorted[dec4.idx]
  
  
  dec1.labs <- labels.sorted[dec1.idx]
  dec2.labs <- labels.sorted[dec2.idx]
  dec3.labs <- labels.sorted[dec3.idx]
  dec4.labs <- labels.sorted[dec4.idx]
  
  
  predicted <- vector()
  observed <- vector()
  
  predicted <- 100* c(mean(dec1.resp),mean(dec2.resp),mean(dec3.resp),mean(dec4.resp))
  
  dec1.rates <- 100* ((sum(dec1.labs))/(length(dec1.labs)))
  dec2.rates <- 100* ((sum(dec2.labs))/(length(dec2.labs)))
  dec3.rates <- 100* ((sum(dec3.labs))/(length(dec3.labs)))
  dec4.rates <- 100* ((sum(dec4.labs))/(length(dec4.labs)))
  
  observed <- c(dec1.rates,dec2.rates,dec3.rates,dec4.rates)
  
  calibration <- data.frame(predicted, observed)
  colnames(calibration) <- c('predicted rates', 'observed rates')
  rownames(calibration) <- c('decile 1', 'decile 2', 'decile 3', 'decile 4')
  list.res <- list()
  list.res[[1]] <- calibration
  list.res[[2]] <- probs.sorted
  list.res[[3]] <- labels.sorted
  return(list.res)
}

quartPoints <- function(list.probs, num.folds) {
  point.1 <- vector()
  point.2 <- vector()
  point.3 <- vector()
  point.4 <- vector()
  point.5 <- vector()
  
  for(f in 1:num.folds) {
    deciles <- quantile(list.probs[[f]], probs=seq(0,1,.25))
    point.1[f] <- deciles[1]
    point.2[f] <- deciles[2]
    point.3[f] <- deciles[3]
    point.4[f] <- deciles[4]
    point.5[f] <- deciles[5]
  }
  list.points <- list()
  list.points[[1]] <- point.1
  list.points[[2]] <- point.2
  list.points[[3]] <- point.3
  list.points[[4]] <- point.4
  list.points[[5]] <- point.5
  
  return(list.points)
}

decPoints <- function(list.probs, num.folds) {
  point.1 <- vector()
  point.2 <- vector()
  point.3 <- vector()
  point.4 <- vector()
  point.5 <- vector()
  point.6 <- vector()
  point.7 <- vector()
  point.8 <- vector()
  point.9 <- vector()
  point.10 <- vector()
  point.11 <- vector()
  
  for(f in 1:num.folds) {
    deciles <- quantile(list.probs[[f]], probs=seq(0,1,.1))
    point.1[f] <- deciles[1]
    point.2[f] <- deciles[2]
    point.3[f] <- deciles[3]
    point.4[f] <- deciles[4]
    point.5[f] <- deciles[5]
    point.6[f] <- deciles[6]
    point.7[f] <- deciles[7]
    point.8[f] <- deciles[8]
    point.9[f] <- deciles[9]
    point.10[f] <- deciles[10]
    point.11[f] <- deciles[11]
  }
  list.points <- list()
  list.points[[1]] <- point.1
  list.points[[2]] <- point.2
  list.points[[3]] <- point.3
  list.points[[4]] <- point.4
  list.points[[5]] <- point.5
  list.points[[6]] <- point.6
  list.points[[7]] <- point.7
  list.points[[8]] <- point.8
  list.points[[9]] <- point.9
  list.points[[10]] <- point.10
  list.points[[11]] <- point.11
  
  return(list.points)
}