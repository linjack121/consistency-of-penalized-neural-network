# X: input data of size (n_x, m)
# parameters: list containing parameters, the output of initialization function
# activation: activation function used in the output layer
#     - sigmoid: for binary classification
#     - linear : for regression
forward_propagation <- function(X, parameters, 
                                activation = c("sigmoid","tanh","relu", "linear"))
{
  # Retrive each parameter from the "parameters"
  W1 <- parameters[["W1"]];
  b1 <- parameters[["b1"]];
  W2 <- parameters[["W2"]];
  b2 <- parameters[["b2"]];
  
  # Implement Forward Propagation to Calculate A2
  Z1 <- W1 %*% X + b1;
  # A1 <- tanh(Z1);
  A1 <- switch(activation, 
               sigmoid = {sigmoid(Z1)},
               tanh = {tanh(Z1)},
               relu = {relu(Z1)},
               linear = {linear(Z1)},
               stop("Invalid Activation Function Name!")
  )
  Z2 <- W2 %*% A1 + b2;
  A2 <- Z2
  
  cache <- list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2)
  
  return(cache)
}


# Sigmoid activation function
sigmoid <- function(x)
{
  return((1+exp(-x))^(-1))
}

# Linear activation
linear <- function(x)
{
  return(x)
}
tanh <- function(x){
  return (2/ (1 + exp(-2*x)) -1)
}

# A test function
# setwd("C:/Users/xshen/Desktop/Reasearch Projects/Neural Network Testing/Simulation/")
# X_assess <- readRDS("X_assess.RDS")
# parameters <- readRDS("parameters.RDS")
# cache <- forward_propagation(X_assess, parameters, "sigmoid")

