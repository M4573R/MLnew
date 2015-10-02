function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X is a matrix of m x n dimension(m x 2) here
% Here X's 1st column is of one's and the 2nd column is of populations
% y is a m * 1(m dimensional vector)
% theta is a 2 x 1 vector here(considering we are doing linear regression
% of one variable
% Here we evaluate h?(x)=?0+?1x
hypothesis = X*theta;

% now we get error = hypothesis - y 
% y is a m x 1 vector and hypothesis is a m x 1 vector

errors = hypothesis - y;

% square all elements of error individually

squared_errors = errors.^2;

% sum errors from 1 to m

sum_of_squared_errors = sum(squared_errors);

% calculate J

J = 1/(2*m)*sum_of_squared_errors;

% =========================================================================

end
