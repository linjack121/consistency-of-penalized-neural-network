# Define three variables
# - n_x: the size of the input layer
# - n_h: the size of the hidden layer
# - n_y: the size of the output layer
# X: input dataset of shape (input size, number of samples)
# Y: labels of shape (output size, number of samples)
layer_sizes <- function(X, Y, n_h)
{
  n_x <- dim(X)[1];
  n_h <- n_h;
  if(is.null(dim(Y)))
  {
    n_y <- 1
  }else
  {
    n_y <- dim(Y)[1];
  }
  
  
  return(list(n_x = n_x, n_h = n_h, n_y = n_y))
}


# A test function
layer_size_test_case <- function()
{
  set.seed(1)
  X_assess <- matrix(rnorm(15), 5, 3);
  Y_assess <- matrix(rnorm(6), 2, 3);
  n_h <- 4;
  
  return(list(X_assess = X_assess, Y_assess = Y_assess, n_h = n_h))
}