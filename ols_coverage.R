library(optparse)
set.seed(42)

for (dims in c(2)){
  for (modl in c('lm')){#,'logistic')){
    # Parse command line arguments
    option_list = list(
      make_option(c('-m', '--model'), action = 'store', default = modl, type = 'character',
                  help = 'regression type, linear or logistic'),
      make_option(c('-d', '--dimension'), action = 'store', default = dims, type = 'numeric',
                  help = 'dimension of the parameter'),
      make_option(c('-r', '--num.repeat'), action = 'store', default = 100, type = 'numeric',
                  help = 'number of repeats'),
      make_option(c('-t', '--theta'), action = 'store', default = 'null', type = 'character',
                  help = 'type of true theta'),
      make_option(c('-n', '--sample.size'), action = 'store', default = 1e5, type = 'numeric',
                  help = 'total sample size')
    )
    opt <- parse_args(OptionParser(option_list = option_list))
  
    # Set up parameters
    model <- opt$model
    d <- opt$dimension
    N <- opt$sample.size
    
    if (opt$theta == 'null') {
      theta.star <- rep(0, d)
    } else if (opt$theta == 'dense') {
      theta.star <- rep(1/sqrt(d), d)
    } else if (opt$theta == 'sparse') {
      theta.star <- c(rep(1, ceiling(d/10)), rep(0, d - ceiling(d/10)))
      theta.star <- theta.star / sqrt(sum(theta.star^2))
    } else {
      theta.star <- seq(0, 1, length = d)
    }
    
    set.seed(42)
    x <- matrix(rnorm(N * d), N, d)
    
    if (model == "lm") {
      y <- as.numeric(x %*% theta.star) + rnorm(N, 0, 1)
    } else {
      y <- ifelse(runif(N, 0, 1) > 1 / (1 + exp(-as.numeric(x %*% theta.star))), -1, 1)
    }
    num.xnew <- 100
    xnew <- matrix(rnorm(num.xnew * d), num.xnew, d)
    ynew <- as.numeric(xnew %*% theta.star)
    
    ols <- function(x, y) {
      xtx <- t(x) %*% x
      xty <- t(x) %*% y
      b_hat <- solve(xtx) %*% xty
      y_hat <- x %*% b_hat
      residuals <- y - y_hat
      s_hat <- sqrt(sum(residuals^2) / (length(y) - ncol(x)))
      list(b = b_hat, s_hat = s_hat, y_hat = y_hat)
    }
    
    lm <- ols(x, y)
    
    predict_ols <- function(lm, newx, ynew, alpha = 0.05) {
      b_hat <- lm$b
      s_hat <- lm$s_hat
      start_time <- Sys.time()
      preds <- newx %*% b_hat
      timing <- as.numeric(Sys.time() - start_time, units = 'secs')
      residuals <- ynew - preds
      mse <- mean(residuals^2)
      n <- nrow(newx)
      se <- sqrt(mse / n)
      margin <- qt(1 - alpha / 2, df = N - ncol(x)) * se
      estimate <- mean(preds)
      upper <- estimate + margin
      lower <- estimate - margin
      ci.length <- upper-lower
      times = timing
      list(estimate = estimate, ci.length = ci.length, resids = residuals, preds = preds, times = timing)
    }
    
    # Compute confidence intervals
    record <- predict_ols(lm, xnew, ynew)
    
    filename <- paste(model, 'd', d, 'theta', opt$theta, sep = '_')
    save(record, file = paste0('sim_data/ols_', filename, '.RData'))
  }
}