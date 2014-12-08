% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

function p = sigmod(z)
    p = zeros(size(z));
    
    p = 1 ./ (1+exp(-z));
    
