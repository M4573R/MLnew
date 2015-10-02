function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % X is a matrix of m x n dimension.
    % y is a m * 1(m dimensional vector)
    % theta is a n x 1 vector here(because it's linear regression of
    % multiple variables). Here we evaluate h(X)
    hypothesis = X*theta;

    % now we get error = hypothesis - y 
    % y is a m x 1 vector and hypothesis is a m x 1 vector

    errors = hypothesis - y;
    
    % here we update theta using the rule of gradient descent.
    %The change in theta (the "gradient") is the sum of the product of X 
    %and the "errors vector", scaled by alpha and 1/m. 
    %Since X is (m x n), and the error vector is (m x 1), 
    %and the expected result is the same size as theta (which is (n x 1), 
    %we need to transpose X before multiplying it by the error vector.
    %Vector multiplication will automatically take care of the inner sum.
 
    theta = theta - alpha*(1/m*(X'*errors)) ;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
