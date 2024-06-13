# Simulations to compare HiGrad confidence intervals' coverage and lengths

library(higrad)
library(optparse)
set.seed(42)

predict.higrad <- function (object, newx, alpha = 0.05, type = "link", prediction.interval = FALSE, 
                            ...) 
{
  if (is.vector(newx)) {
    newx = matrix(newx, 1, length(newx))
  }
  if (ncol(newx) != length(object$coefficients)) {
    stop("'newx' has the wrong dimension")
  }
  Sigma0 <- object$Sigma0
  B <- nrow(Sigma0)
  mu <- as.numeric(newx %*% object$coefficients)
  mus <- newx %*% t(object$coefficients.bootstrap)
  ses <- sqrt(sum(Sigma0) * colSums(t(mus - mu) * solve(Sigma0, 
                                                        t(mus - mu)))/(B^2 * (B - 1)))
  if (prediction.interval) {
    margin <- qt(1 - alpha/2, B - 1) * ses * sqrt(2)
  }
  else {
    margin <- qt(1 - alpha/2, B - 1) * ses
  }
  upper <- mu + margin
  lower <- mu - margin
  if (object$model == "logistic" & type == "response") {
    mu = 1/(1 + exp(-mu))
    upper = 1/(1 + exp(-upper))
    lower = 1/(1 + exp(-lower))
  }
  out <- list(pred = mu, upper = upper, lower = lower, ycalcs = mus)
  return(out)
}

set.seed(42)

for (dims in c(500)){
  for (modl in c('lm','logistic')){
    # parse command line argument
    option_list = list(
      make_option(c("-m", "--model"), action = "store", default = modl, type = 'character',
                  help = "regression type, linear or logistic"),
      make_option(c("-d", "--dimension"), action = "store", default = dims, type = 'numeric', # 50
                  help = "dimension of the parameter"),
      make_option(c("-r", "--num.repeat"), action = "store", default = 100, type = 'numeric', # 100
                  help = "number of repeats"),
      make_option(c("-t", "--theta"), action = "store", default = "dense", type = "character",
                  help = "type of true theta"),
      make_option(c("-n", "--sample.size"), action = "store", default = 5e3, type = 'numeric', # 1e5
                  help = "total sample size")  
    )
    opt <- parse_args(OptionParser(option_list=option_list))
    opt$theta <- 'null'
    
    # set up parameters
    model <- opt$model
    d <- opt$dimension
    N <- opt$sample.size
    num.repeat <- opt$num.repeat
    if (opt$theta == "null") {
      theta.star <- rep(0, d)
    } else if (opt$theta == "dense") {
      theta.star <- rep(1/sqrt(d), d)
    } else if (opt$theta == "sparse") {
      theta.star <- c(rep(1, ceiling(d/10)), rep(0, d - ceiling(d/10)))
      theta.star <- theta.star / sqrt(sum(theta.star^2))
    } else {
      theta.star <- seq(0, 1, length = d)
    }
    
    sigma <- 1
    alpha <- 0.55
    eta <- ifelse(model == "lm", 0.1, 0.4)
    burnin <- 0
    
    # create new samples for prediction
    num.xnew <- 100
    xnew <- matrix(rnorm(num.xnew * d), num.xnew, d)
    ynew <- as.numeric(xnew %*% theta.star) 
    
    filename <- paste(model, 'd', d, 'theta', opt$theta, sep = '_')
    
    # print simulation type
    cat("Regression type:", model, "\n")
    cat("Dimension:", d, "\n")
    cat("Starting simulation...\n")
    
    # create configurations
    configs <- list()
    configs[[length(configs) + 1]] <- list(nsplits = 2, nthreads = 2, step.ratio = 1, n0 = NA, eta = eta, burnin = burnin, alpha = alpha)
    
    # create record list
    record <- list(length(configs))
    for (j in 1:length(configs)) {
      record[[j]] <- list(pred = matrix(0, num.repeat, num.xnew), 
                          ci.length = matrix(0, num.repeat, num.xnew), 
                          coverage = matrix(0, num.repeat, num.xnew),
                          estimate = matrix(0, num.repeat, d),
                          times = numeric(num.repeat),
                          ycalcs = numeric(num.repeat))
    }
    
    # start simulation
    for (i in 1:num.repeat) {
      # set.seed(i)
      x <- matrix(rnorm(N * d), N, d)
      if (model == "lm") {
        y <- as.numeric(x %*% theta.star) + rnorm(N, 0, sigma)
      } else {
        y <- ifelse(runif(N, 0, 1) > 1 / (1 + exp(-as.numeric(x %*% theta.star))), -1, 1)
      }
      
      for (j in 1:length(configs)) {
        start_time <- Sys.time()
        fit <- higrad(x, y, model = model, 
                      nsplits = configs[[j]]$nsplits, 
                      nthreads = configs[[j]]$nthreads, 
                      step.ratio = configs[[j]]$step.ratio,
                      n0 = configs[[j]]$n0,
                      burnin = configs[[j]]$burnin,
                      eta = configs[[j]]$eta, 
                      alpha = configs[[j]]$alpha)
        timing <- as.numeric(Sys.time() - start_time, units = 'secs')
        pred <- predict(fit, newx = xnew, alpha = 0.1)
        record[[j]]$pred[i, ] <- pred$pred
        record[[j]]$ci.length[i, ] <- pred$upper - pred$lower
        record[[j]]$coverage[i, ] <- (ynew >= pred$lower) & (ynew <= pred$upper)
        record[[j]]$estimate[i, ] <- fit$coefficients
        record[[j]]$times[i] <- timing
        record[[j]]$ycalcs[i] <- pred$ycalcs
      }
    }
    filename <- paste(model, 'd', d, 'theta', opt$theta, sep = '_')
    save(record, file = paste0('sim_data/higrad_', filename, '.RData'))
  }
}