function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

    function [z2_, a2_, z3_, a3_] = fwProp(a1_, Theta1, Theta2)
        %Catenating bias value for layer 1 and plugging it in the sigmoid
        a1Bias_ = [1; a1_];
        z2_ = Theta1 * a1Bias_;
        a2_ = sigmoid(z2_);
        %Catenating bias value for layer 1 and plugging it in the sigmoid
        a2Bias_ = [1; a2_];
        z3_ = Theta2 * a2Bias_;
        a3_ = sigmoid(z3_);
    end

for i = 1:m
    [~, ~, ~, hTheta_] = fwProp(X(i, :)', Theta1, Theta2);    
    J = J + (-full(ind2vec(y(i), num_labels))' * log(hTheta_) ...
        - (1 - full(ind2vec(y(i), num_labels))') * log(1 - hTheta_));
end
J = J/m ;

Theta1NoBias = Theta1(:, 2:end);
theta1NoBiasUnrolled_ = Theta1NoBias(:);
Theta2NoBias = Theta2(:, 2:end);
theta2NoBiasUnrolled_ = Theta2NoBias(:);

J = J + (lambda/(2*m)) ...
    * (sum(theta1NoBiasUnrolled_ .^ 2) + sum(theta2NoBiasUnrolled_ .^ 2));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for i = 1:m
    a1_ = X(i, :)'; % (400 x 1)
    [z2_, a2_, ~, a3_] = fwProp(a1_, Theta1, Theta2);
    d3_ = a3_ - full(ind2vec(y(i), num_labels)); % (10 x 1)
    d2Bias_ = (Theta2' * d3_) .* [0; sigmoidGradient(z2_)]; % (26 x 1)
    d2_ = d2Bias_(2:end); % (25 x 1)
    
    Delta1 = Delta1 + d2_ * [1; a1_]'; % (25 x 401)
    Delta2 = Delta2 + d3_ * [1; a2_]'; % (10 x 26)
end

Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

%Adding regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
