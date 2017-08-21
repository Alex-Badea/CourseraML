function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
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
Theta = reshape(nn_params, num_labels, input_layer_size + 1);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta_grad = zeros(size(Theta));

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

    function [z2_, a2_] = fwProp(a1_, Theta)
        %Catenating bias value for layer 1 and plugging it in the sigmoid
        a1Bias_ = [1; a1_];
        z2_ = Theta * a1Bias_;
        a2_ = sigmoid(z2_);
    end

for i = 1:m
    [~, hTheta_] = fwProp(X(i, :)', Theta);    
    J = J + (-full(ind2vec(y(i), num_labels))' * log(hTheta_) ...
        - (1 - full(ind2vec(y(i), num_labels))') * log(1 - hTheta_));
end
J = J/m ;

ThetaNoBias = Theta(:, 2:end);
thetaNoBiasUnrolled_ = ThetaNoBias(:);

J = J + (lambda/(2*m)) ...
    * (sum(thetaNoBiasUnrolled_ .^ 2));

Delta = zeros(size(Theta));
for i = 1:m
    a1_ = X(i, :)'; % (400 x 1)
    [~, a2_] = fwProp(a1_, Theta);
    d2_ = a2_ - full(ind2vec(y(i), num_labels)); % (10 x 1)
    
    Delta = Delta + d2_ * [1; a1_]'; % (10 x 401)
end

Theta_grad = Delta./m;

%Adding regularization
Theta_grad(:, 2:end) = Theta_grad(:, 2:end) + (lambda/m) * Theta(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = Theta_grad(:);


end
