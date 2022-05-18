# Back propagation
# parameters: list containing parameters
# cache: list containg Z1, A1, Z2, A2
# X: input data
# Y: true response vector
# lambda: regularization parameter, scalar
back_propagation <- function(parameters, cache, X, Y, lambda,
                             activation = c("sigmoid", "tanh","linear","relu"))
{
  m <- dim(X)[2];
  
  # Retrive W1 and W2 from parameters
  W1 <- parameters[["W1"]];
  W2 <- parameters[["W2"]];
  
  # Retrieve A1 and A2 from cache
  A1 <- cache[["A1"]];
  A2 <- cache[["A2"]];
  
  # Back propagation
  if(activation == "sigmoid")
  {
    dZ2 <- A2 - Y;
    dW2 <- (1/m) * tcrossprod(dZ2, A1) + 2*lambda/m * W2;
    db2 <- (1/m) * apply(dZ2, 1, sum);
    # dZ1 <- crossprod(W2, dZ2) * (1 - A1^2);
    dZ1 <- crossprod(W2, dZ2) * A1 * (1-A1);
    dW1 <- (1/m) * tcrossprod(dZ1, X) + lambda/m * W1;
    db1 <- (1/m) * apply(dZ1, 1, sum);
  }
  if(activation == "tanh")
  {
    dZ2 <- A2 - Y;
    dW2 <- (1/m) * tcrossprod(dZ2, A1) + 2*lambda/m * W2;
    db2 <- (1/m) * apply(dZ2, 1, sum);
    # dZ1 <- crossprod(W2, dZ2) * (1 - A1^2);
    dZ1 <- crossprod(W2, dZ2) * (1+A1) * (1-A1);
    dW1 <- (1/m) * tcrossprod(dZ1, X) + 2*lambda/m * W1;
    db1 <- (1/m) * apply(dZ1, 1, sum);
  }
  
  if(activation == "relu")
  {
    dZ2 <- A2 - Y;
    dW2 <- (1/m) * tcrossprod(dZ2, A1) + 2*lambda/m * W2
    db2 <- (1/m) * apply(dZ2, 1, sum);
    
   
    dZ1 <- crossprod(W2, dZ2) * ifelse(A1>0,1,0);#relu
   
    dW1 <- (1/m) * tcrossprod(dZ1, X) + 2*lambda/m * W1
    db1 <- (1/m) * apply(dZ1, 1, sum)
    
    
  }
  
  if(activation == "linear")
  {
    dZ2 <- 2 * (A2 - Y);
    dW2 <- (1/m) * tcrossprod(dZ2, A1) + lambda/m * W2;
    db2 <- (1/m) * apply(dZ2, 1, sum);
    dZ1 <- crossprod(W2, dZ2) * A1 * (1-A1);
    dW1 <- (1/m) * tcrossprod(dZ1, X) + lambda/m * W1;
    db1 <- (1/m) * apply(dZ1, 1, sum);
  }
  
  grads <- list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2);
  
  return(grads)
  
}



