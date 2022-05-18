source("layer_sizes.R")
source("initialize_parameters.R")
source("forward_propagation.R")
source("compute_cost.R")
source("back_propagation.R")
source("update_parameters.R")




# Combining altogether
nn_model <- function(X, Y, n_h, lambda = 0.1, learning_rate = 1.2, activation = "tanh",num_iterations = 1e5, print_cost = FALSE)
{
  set.seed(3)
  n_x <- layer_sizes(X, Y, n_h)[[1]];
  n_y <- layer_sizes(X, Y, n_h)[[3]];
  
  # Initialize parameters, then retrieve W1, b1, W2, b2. 
  # Inputs: n_x, n_h, n_y.
  # Outputs: W1, b2, W2, b2, parameters
  parameters <- initialize_parameters(n_x, n_h, n_y);
  W1 <- parameters[["W1"]];
  b1 <- parameters[["b1"]];
  W2 <- parameters[["W2"]];
  b2 <- parameters[["b2"]];
  
  # Loop (gradient descent)
  for(i in 1:num_iterations)
  {
    # Forward propagation.
    # Inputs: X, parameters
    # Outputs: A2, cache
    cache <- forward_propagation(X, parameters, activation);
    A2 <- cache[["A2"]];
    
    # Cost Function
    # Inputs: A2, Y, parameters
    # Outputs: cost
    cost <- compute_cost(A2, Y, parameters, lambda, activation);
    
    # Back propagation
    # Inputs: parameters, cache, X, Y
    # Outputs: grads
    grads = back_propagation(parameters, cache, X, Y, lambda, activation);
    
    # Gradient descent parameter update
    # Inputs: parameters, grads
    # Outputs: parameters
    parameters <- update_parameters(parameters, grads, learning_rate)
    
    #if(print_cost && (i %% 1000 == 0))
    #{
    #  cat("Cost after iteration ", i, ", ", cost, "\n")
    #}
  }
  #MSE = mean((Y-Y_true)^2)
  #return(list(parameters,Y_pred = Y, Y_true = Y_true, MSE = MSE))
  returnList <- list(parameters = parameters, cost = cost)
  print("check")
  return(returnList)
  
}
#res <- nn_model(X=X,Y=Y,n_h =3, lambda = 0.05, learning_rate = 0.001,activation = "tanh",num_iterations = 1e4, print_cost = TRUE)


# A test function
# setwd("C:/Users/xshen/Desktop/Reasearch Projects/Neural Network Testing/Simulation/")
# X_assess <- readRDS("X_assess.RDS")
# Y_assess <- readRDS("Y_assess.RDS") 
# parameters <- nn_model(X_assess, Y_assess, 4, "sigmoid")
