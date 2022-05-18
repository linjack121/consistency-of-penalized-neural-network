# Caculate the cost function J
# For sigmoid activation, i.e., the binary classification, the cross-entropy is used
# For linear activation, i.e., the regression, the squared error loss is used
compute_cost <- function(A2, Y,
                         parameters, lambda,
                         activation = c("sigmoid","tanh", "linear"))
{
  m <- dim(Y)[2];    # number of samples
  W1 <- parameters[["W1"]];
  W2 <- parameters[["W2"]];
  
  L2_regularization_cost <- lambda/(2*m) * (sum(W1^2) + sum(W2^2))
  
  if(activation == "sigmoid")
  {
    logprobs <- Y * log(A2) + (1 - Y) * log(1 - A2);
    cost <- -(1/m) * sum(logprobs);
    cost <- cost + L2_regularization_cost;
  }
  
  if(activation == "linear")
  {
    sqerrors <- (Y - A2)^2;
    cost <- (1/m) * sum(sqerrors);
    cost <- cost + L2_regularization_cost;
  }
  
  if(activation == "tanh")
  {
    sqerrors <- (Y - A2)^2;
    cost <- (1/m) * sum(sqerrors);
    cost <- cost + L2_regularization_cost;
  }
  
  if(activation == "relu")
  {
    sqerrors <- (Y - A2)^2;
    cost <- (1/m) * sum(sqerrors);
    cost <- cost + L2_regularization_cost;
  }
  
  cost <- as.numeric(cost)
  
  return(cost)
  
}


# A test function
# setwd("C:/Users/xshen/Desktop/Reasearch Projects/Neural Network Testing/Simulation/")
# Y_assess <- readRDS("Y_assess.RDS")
# parameters <- readRDS("parameters.RDS")
# A2 <- matrix(c(0.5002307 ,  0.49985831,  0.50023963),1,3)
# compute_cost(A2, Y_assess, "sigmoid")
