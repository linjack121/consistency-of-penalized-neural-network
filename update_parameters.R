# Update parameters Using gradient descent
update_parameters <- function(parameters, grads, learning_rate = 1.2)
{
  # Retrieve each parameter
  W1 <- parameters[["W1"]];
  b1 <- parameters[["b1"]];
  W2 <- parameters[["W2"]];
  b2 <- parameters[["b2"]];
  
  # Retrive each gradient
  dW1 <- grads[["dW1"]];
  db1 <- grads[["db1"]];
  dW2 <- grads[["dW2"]];
  db2 <- grads[["db2"]];
  
  # Update rule for each parameter
  W1 <- W1 - learning_rate * dW1;
  b1 <- b1 - learning_rate * db1;
  W2 <- W2 - learning_rate * dW2;
  b2 <- b2 - learning_rate * db2;
  
  parameters <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2);
  
  return(parameters)
  
}


# A test function
# W1 <- matrix(c(-0.00615039,  0.0169021, -0.02311792,  0.03137121, -0.0169217 , -0.01752545, 0.00935436, -0.05018221), 4, 2, byrow = T);
# W2 <- c(-0.0104319 , -0.04019007,  0.01607211,  0.04440255);
# b1 <- matrix(c(-8.97523455e-07, 8.15562092e-06, 6.04810633e-07, -2.54560700e-06), 4,1);
# b2 <- 9.14954378e-05;
# parameters <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2);
# 
# dW1 <- matrix(c(0.00023322, -0.00205423, 0.00082222, -0.00700776, -0.00031831,  0.0028636, -0.00092857,  0.00809933), 4, 2, byrow = T);
# dW2 <- c(-1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03);
# db1 <- matrix(c(1.05570087e-07, -3.81814487e-06, -1.90155145e-07, 5.46467802e-07),4,1);
# db2 <- -1.08923140e-05;
# grads <- list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2);
# 
# parameters <- update_parameters(parameters, grads)
