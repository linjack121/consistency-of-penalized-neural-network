# load all necessary sources
setwd("C:/Users/jlin/Desktop/consistency of PNN")
source("layer_sizes.R")
source("initialize_parameters.R")
source("forward_propagation.R")
source("compute_cost.R")
source("back_propagation.R")
source("update_parameters.R")
source("predict.R")
source("nn_model.R")



###########################################
# Simulation1:
# Consistency of NN Sieve Estimators
###########################################
# Part 1: Truth is a neural network
# Generate True model 
set.seed(1)

nN <- 5000; # number of samples
n_h_true <- 2;
n_x_true <- 1;
n_y_true <- 1;
#X_true <- matrix(rnorm(nN * n_x_true), n_x_true, nN);
X_true <- matrix(runif(nN * n_x_true,-2,2), n_x_true, nN);
W1 <- matrix(c(2,-1), n_h_true, n_x_true); #matrix(rnorm(n_h_true*n_x_true), n_h_true, n_x_true);
b1 <- rep(1, n_h_true); #rnorm(n_h_true);
W2 <- matrix(c(-1,1), n_y_true, n_h_true); #matrix(rnorm(n_y_true*n_h_true), n_y_true, n_h_true);
b2 <- -1; #rnorm(n_y_true);
parameters_true <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2);
cache_true <- forward_propagation(X_true, parameters_true, activation = "sigmoid");
Y_true <- cache_true[["A2"]];

n_h <- 10;
# Fitted model
n_sample <- 100*c( 1 ,2, 5, 10,20);
error <- rep(0, length(n_sample));
cost <- rep(0, length(n_sample));
Qnf0 <- rep(0, length(n_sample));

for(j in 1:length(n_sample))
{
  index <- sample(nN, n_sample[j]); # choose part of samples randomly
  Y_true_sample <- Y_true[, index];
  X <- matrix(X_true[, index], n_x_true, n_sample[j]);
  Y <- matrix(Y_true_sample + rnorm(n_sample[j],0, 0.7), n_y_true, n_sample[j]);
  
  
  
    result_List <- nn_model(X, Y, n_h, lambda = 0.5,activation = "tanh",
                                             learning_rate = 0.005, num_iterations = 20000, print_cost = TRUE);
    parameters_fitted <- result_List$parameters;
    # parameters_fitted <- nn_model(X, Y, n_h[i], lambda = 0,  activation = "linear",
    #                               learning_rate = 0.1, num_iterations = 3e4, print_cost = TRUE);
    cache_fitted <- forward_propagation(X, parameters_fitted, activation = "tanh");
    Y_fitted <- cache_fitted[["A2"]];
  
  
  error[j] <- (1/n_sample[j]) * sum((Y_fitted - Y_true_sample)^2);
  cost[j] <- result_List$cost;
  Qnf0[j] <- (1/n_sample[j]) * sum((Y - Y_true_sample)^2);
  cat("Cost after iteration ", j, ", ", error[j], "\n")
  cat("Qnf0 after iteration ", j, ", ", Qnf0[j], "\n")
  
  if(j == 1)
  {
    plot(X, Y_true_sample, xlim = range(X), ylim = range(Y), xlab = "x", ylab = "y", pch = 8, cex = 0.3,
         main = paste0("True Function vs Fitted Functions"))
    lines(X_true[order(X_true)], Y_true[order(X_true)], xlim = range(X), ylim = range(Y_true), pch = 8, col = j)
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 2, col = j+1)
  }else
  {
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 1+j, col = j+1)
  }
  
}

legend("topright", legend = c("Truth", n_sample), lty = 1:(length(n_sample)+1), 
       col = 1:(length(n_sample)+1), cex = 0.8)


# Part 2: Truth is some continuous function
# Generate True model 
########################################
# Truth = sine function
########################################
set.seed(1)

nN <- 5000; # number of samples
n_x_true <- 1;
n_y_true <- 1;
#X_true <- matrix(rnorm(nN * n_x_true), n_x_true, nN);
X_true <- matrix(runif(nN * n_x_true,-2,2), n_x_true, nN);
Y_true <- sin((pi/3) * X_true) + 1/3 * cos((pi/4) * X_true + 1);

#Y_true <- sin(X_true)+2*exp((-16)*X_true^2)
# Fitted model
n_sample <- 100*c( 1, 2,5, 10,20);
error <- rep(0, length(n_sample));
cost <- rep(0, length(n_sample));
Qnf0 <- rep(0, length(n_sample));
n_h <- 10
for(j in 1:length(n_sample))
{
  index <- sample(nN, n_sample[j]);
  Y_true_sample <- Y_true[, index];
  X <- matrix(X_true[, index], n_x_true, n_sample[j]);
  Y <- matrix(Y_true_sample + rnorm(n_sample[j],0, 0.7), n_y_true, n_sample[j]);
  

  
    result_List <- nn_model(X, Y, n_h, lambda = 0.01,activation = "tanh",
                            learning_rate = 0.005, num_iterations = 20000, print_cost = TRUE);
    parameters_fitted <- result_List$parameters;
    # parameters_fitted <- nn_model(X, Y, n_h[i], lambda = 0,  activation = "linear",
    #                               learning_rate = 0.1, num_iterations = 2e4, print_cost = TRUE);
    cache_fitted <- forward_propagation(X, parameters_fitted, activation = "sigmoid");
    Y_fitted <- cache_fitted[["A2"]];

  
  error[j] <- (1/n_sample[j]) * sum((Y_fitted - Y_true_sample)^2);
  cost[j] <- result_List$cost;
  Qnf0[j] <- (1/n_sample[j]) * sum((Y - Y_true_sample)^2);
  cat("Cost after iteration ", j, ", ", error[j], "\n")
  cat("Qnf0 after iteration ", j, ", ", Qnf0[j], "\n")
  if(j == 1)
  {
    plot(X, Y_true_sample, xlim = range(X), ylim = range(Y), xlab = "x", ylab = "y", pch = 16, cex = 0.3,
         main = paste0("True Function vs Fitted Functions"))
    lines(X_true[order(X_true)], Y_true[order(X_true)], xlim = range(X), ylim = range(Y_true), pch = 16, col = j)
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 2, col = j+1)
  }else
  {
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 1+j, col = j+1)
  }
  
}

legend("topleft", legend = c("Truth", n_sample), lty = 1:(length(n_sample)+1),
       col = 1:(length(n_sample)+1), cex = 0.8)



########################################
# Truth = piecewise continuous function or sin + exp
#######################################
set.seed(1)

piecewise_fun <- function(x)
{
  y <- -2*x;
  n <- length(x);
  for(i in 1:n)
  {
    if(x[i] > 0)
    {
      y[i] <- x[i]^(1/2) * (x[i]-1/4);
    }
  }
  
  return(y)
}

nN <- 5000; # number of samples
n_x_true <- 1;
n_y_true <- 1;
#X_true <- matrix(rnorm(nN * n_x_true), n_x_true, nN);
X_true <- matrix(runif(nN * n_x_true,-2,2), n_x_true, nN);
Y_true <- sin(X_true)+2*exp((-16)*X_true^2)
#Y_true <- piecewise_fun(X_true)
n_h = 10
# Fitted model
n_sample <- 100*c( 1,2, 5, 10,20);
error <- rep(0, length(n_sample));
cost <- rep(0, length(n_sample));
Qnf0 <- rep(0, length(n_sample));

for(j in 1:length(n_sample))
{
  index <- sample(nN, n_sample[j]);
  Y_true_sample <- Y_true[, index];
  X <- matrix(X_true[, index], n_x_true, n_sample[j]);
  Y <- matrix(Y_true_sample + rnorm(n_sample[j],0, 0.7), n_y_true, n_sample[j]);
  
  n_h <- c(floor(n_sample[j] ^ (1/4)));
  upper_bound <- 10 * n_sample[j] ^ (1/4);
  
  
    result_List <- nn_model(X, Y, n_h, lambda = 0.01,activation = "tanh",
                            learning_rate = 0.005, num_iterations = 30000, print_cost = TRUE);
    parameters_fitted <- result_List$parameters;
    # parameters_fitted <- nn_model(X, Y, n_h[i], lambda = 0,  activation = "linear",
    #                               learning_rate = 0.1, num_iterations = 3e4, print_cost = TRUE);
    cache_fitted <- forward_propagation(X, parameters_fitted, activation = "tanh");
    Y_fitted <- cache_fitted[["A2"]];
  
  
  error[j] <- (1/n_sample[j]) * sum((Y_fitted - Y_true_sample)^2);
  cost[j] <- result_List$cost;
  Qnf0[j] <- (1/n_sample[j]) * sum((Y - Y_true_sample)^2);
  cat("Cost after iteration ", j, ", ", error[j], "\n")
  cat("Qnf0 after iteration ", j, ", ", Qnf0[j], "\n")
  if(j == 1)
  {
    plot(X, Y_true_sample, xlim = range(X), ylim = range(Y), xlab = "x", ylab = "y", pch = 16, cex = 0.3,
         main = paste0("True Function vs Fitted Functions"))
    lines(X_true[order(X_true)], Y_true[order(X_true)], xlim = range(X), ylim = range(Y_true), pch = 16, col = j)
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 2, col = j+1)
  }else
  {
    lines(X[order(X)], Y_fitted[order(X)], lwd = 3, lty = 1+j, col = j+1)
  }
  
}

legend("topleft", legend = c("Truth", n_sample), lty = 1:(length(n_sample)+1), 
       col = 1:(length(n_sample)+1),  cex = 0.8)

