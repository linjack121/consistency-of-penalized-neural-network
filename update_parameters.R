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


