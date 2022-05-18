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




