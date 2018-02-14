catToBin <- function(data) {
  bin.names <- levels(data)
  bin.data <- matrix(0, nrow=length(data), ncol=length(bin.names))
  for(idx in 1:length(data)){
    col.idx <- which(bin.names == data[idx])
    bin.data[idx,col.idx] <- 1
  }
  
  bin.data <- data.frame(bin.data)
  colnames(bin.data) <- bin.names
  
  return(bin.data)
}