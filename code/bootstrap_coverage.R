library(optparse)

set.seed(42)
for (dims in c(2)){
  for (modl in c('lm','logistic')){
  # Parse command line arguments
  option_list <- list(
    make_option(c("-m", "--model"), action = "store", default = modl, type = 'character',
                help = "regression type, linear or logistic"),
    make_option(c("-d", "--dimension"), action = "store", default = dims, type = 'numeric',
                help = "dimension of the parameter"),
    make_option(c("-r", "--num.repeat"), action = "store", default = 100, type = 'numeric',
                help = "number of repeats"),
    make_option(c("-t", "--theta"), action = "store", default = "null", type = 'character',
                help = "type of true theta"),
    make_option(c("-n", "--sample.size"), action = "store", default = 5e3, type = 'numeric',
                help = "total sample size")  
  )
  opt <- parse_args(OptionParser(option_list = option_list))
  
  opt$theta <- 'null'
  
  # Set up parameters
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
  
  # Create new samples for prediction
  num.xnew <- 100
  xnew <- matrix(rnorm(num.xnew * d), num.xnew, d)
  ynew <- as.numeric(xnew %*% theta.star)
  
  filename <- paste(model, 'd', d, 'theta', opt$theta, sep = '_')
  
  # Print simulation type
  cat("Regression type:", model, "\n")
  cat("Dimension:", d, "\n")
  cat("Starting simulation...\n")
  
  # Create configurations
  configs <- list()
  configs[[length(configs) + 1]] <- list(step.ratio = 1, eta = eta, burnin = burnin, alpha = alpha)
  
  # Create record list
  record <- list()
  for (j in 1:length(configs)) {
    record[[j]] <- list(pred = matrix(0, num.repeat, num.xnew), 
                        ci.length = matrix(0, num.repeat, num.xnew), 
                        coverage = matrix(0, num.repeat, num.xnew),
                        estimate = matrix(0, num.repeat, d),
                        times = numeric(num.repeat),
                        ycalcs = numeric(num.repeat))
  }
  
  # Stochastic Gradient Descent function with bootstrapping
  sgd <- function(x, y, learning_rate, epochs) {
    n <- nrow(x)
    d <- ncol(x)
    theta_history <- matrix(0, nrow = epochs, ncol = d)
    
    theta <- rep(0, d) # Initialize theta to zero
    
    for (epoch in 1:epochs) {
      theta_epoch <- matrix(0, nrow = n, ncol = d)
      
      for (i in 1:n) {
        idx <- sample(1:n, 1)
        xi <- x[idx, , drop = FALSE]
        yi <- y[idx]
        gradient <- t(xi) * (as.numeric(xi %*% theta) - yi)
        theta <- theta - learning_rate * gradient
        theta_epoch[i, ] <- theta
      }
      
      theta_history[epoch, ] <- colMeans(theta_epoch) # Average theta values within this epoch
    }
    
    out <- list()
    out$model <- model
    out$coefficients <- colMeans(theta_history)
    out$coefficients.bootstrap <- theta_history
    class(out) <- "avgsgd"
    return(out)
  }
  
  # Prediction function for avgsgd
  predict.avgsgd <- function(object, newx, alpha = 0.1, type = "link", prediction.interval = FALSE, ...) {
    if (is.vector(newx)) {
      newx <- matrix(newx, 1, length(newx))
    }
    if (ncol(newx) != length(object$coefficients)) {
      stop("'newx' has the wrong dimension")
    }
    
    mu <- as.numeric(newx %*% object$coefficients)
    mus <- newx %*% t(object$coefficients.bootstrap)
    B <- nrow(object$coefficients.bootstrap)
    ses <- apply(mus, 1, sd)
    
    if (prediction.interval) {
      margin <- qt(1 - alpha/2, B - 1) * ses * sqrt(2)
    } else {
      margin <- qt(1 - alpha/2, B - 1) * ses
    }
    upper <- mu + margin
    lower <- mu - margin
    
    if (object$model == "logistic" & type == "response") {
      mu <- 1/(1 + exp(-mu))
      if (!is.null(object$coefficients.bootstrap)) {
        upper <- 1/(1 + exp(-upper))
        lower <- 1/(1 + exp(-lower))
      }
    }
    
    out <- list(pred = mu, upper = upper, lower = lower, y_calc = mus)
    return(out)
  }
  
  # Start simulation
  for (i in 1:num.repeat) {
    # set.seed(i)
    x <- matrix(rnorm(N * d), N, d)
    if (model == "lm") {
      y <- as.numeric(x %*% theta.star) + rnorm(N, 0, sigma)
    } else {
      y <- ifelse(runif(N, 0, 1) > 1 / (1 + exp(-as.numeric(x %*% theta.star))), -1, 1)
    }
    
    for (j in 1:length(configs)) {
      start_time <- Sys.time() # Start timing
      fit <- sgd(x, y, learning_rate = configs[[j]]$eta, epochs = 4)
      timing <- as.numeric(Sys.time() - start_time, units = 'secs')
      
      pred <- predict(fit, newx = xnew, alpha = 0.1)
      record[[j]]$pred[i, ] <- pred$pred
      record[[j]]$ci.length[i, ] <- pred$upper - pred$lower
      record[[j]]$coverage[i, ] <- (ynew >= pred$lower) & (ynew <= pred$upper)
      record[[j]]$estimate[i, ] <- fit$coefficients
      record[[j]]$times[i] <- timing
      record[[j]]$ycalcs[i] <- pred$y_calc 
    }
  }
  
  filename <- paste(model, 'd', d, 'theta', opt$theta, sep = '_')
  save(record, file = paste0('sim_data/boot_', filename, '.RData'))
  }
}
