source("forward_propagation.R")

# Prediction
predict <- function(parameters, X, activation)
{
  cache <- forward_propagation(X, parameters, activation);
  A2 <- cache[["A2"]];
  
  if(activation == "sigmoid")
  {
    predictions <- as.numeric(A2 > 0.5);
  }
  
  if(activation == "linear")
  {
    predictions <- A2;
  }
  
  if(activation == "linear")
  {
    predictions <- A2;
  }
  return(predictions)
}


# A test function
# setwd("C:/Users/xshen/Desktop/Reasearch Projects/Neural Network Testing/Simulation/")
# X_assess <- readRDS("X_assess.RDS")
# W1 <- matrix(c(-0.00615039,  0.0169021, -0.02311792,  0.03137121, -0.0169217 , -0.01752545, 0.00935436, -0.05018221), 4, 2, byrow = T);
# W2 <- c(-0.0104319 , -0.04019007,  0.01607211,  0.04440255);
# b1 <- c(-8.97523455e-07, 8.15562092e-06, 6.04810633e-07, -2.54560700e-06);
# b2 <- 9.14954378e-05;
# parameters <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2);
# 
# predictions <- predict(parameters, X_assess, "sigmoid")


