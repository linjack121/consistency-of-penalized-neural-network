# Initialize the weights matrices with random values. We use 0.01*N(0,1)
# Initialize the bias vectors as zeros
# W1: weight matrix of shape (n_h, n_x)
# b1: bias vector of shape (n_h, 1)
# W2: weight matrix of shape (n_y, n_h)
# b2: bias vector of shape (n_y, 1)
initialize_parameters <- function(n_x, n_h, n_y)
{
  set.seed(2)
  
  W1 <- matrix(0.1 * rnorm(n_x * n_h), n_h, n_x);
  b1 <- rep(0, n_h);
  W2 <- matrix(0.1 * rnorm(n_h * n_y), n_y, n_h);
  b2 <- rep(0, n_y);
  
  parameters <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2);
  
  return(parameters)
}



# A test function
initialize_parameters_test_case <- function()
{
  n_x <- 2;
  n_h <- 4;
  n_y <- 1;
  
  return(list(n_x = n_x, n_h = n_h, n_y = n_y))
}
