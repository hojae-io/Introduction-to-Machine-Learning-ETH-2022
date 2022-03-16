# Luca Text
In our implementation we utilized sklearn's Ridge method. RidgeCV was the one we initially wanted to use but it limited our freedom to manipulate the cross-validation process.
First we import all the packages and set our constants (folds, lambdas)
After that, we iterate over the different lambdas and switch through the folds 10 times for each one. We split the dataframe using easy pandas iloc notation and use sklearn's KFolds method to give us the relevant indices at every step in the loop. After fitting the Ridge model with the current training data, we test it against our test data and at every step we are meaning the root meaned squared error of the resulting prediction (using sklearn's mean_squared_error)

Afterwards we do some simple plotting and exporting our results to a csv file.

Some steps that we have tried to improve our result are:
1) using a different solver
2) changing the number of folds
3) using the monte-Carlo folding instead of the sequential
4) meaning again over 10^4 iterations, each of which gets a different random Monte-Carlo shuffle.
5) Changing the fit_intercept parameter of our Ridge model to False

The result against the solution was worsened by the steps 1), 2), 3) but was improved significantly by using a combination of 3) and 4) as well as doing 5) (this helped especially much!).

Thank you for reading!
Luca Entremont