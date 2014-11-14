function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
[parms throwaway] = size(theta);

for iter = 1:num_iters

h = sigmoid(X*theta);
slope(1:parms) = 0;
for k = 1:parms
	% For each feature, multiply the feature value of every data point
	% by the difference in the prediction value minus the real value
	mat = X(:,k).*(h - y); 
	% Sum over all data points to get the partial derivative of feature k
	slope(k) = sum(mat);
endfor

% Update vector theta by multiplying a constant by the negative of the slope
theta = theta - alpha/m * transpose(slope);



    % ============================================================
[cost, grad] = costFunction(theta, X, y);
    % Save the cost J in every iteration    
  	cost
	%theta
	%slope'
	%temp = [log(h) log(1-h)];
	%[sum(temp(:,1)) sum(temp(:,2))]'
	
	J_history(iter) = cost;

end

end
