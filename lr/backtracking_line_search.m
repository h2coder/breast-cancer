function [ step ] = backtracking_line_search( step, theta,X,y,lambda,MAX_SEARCH )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
%   Detailed explanation goes here
    if nargin < 6
        MAX_SEARCH = 20 ;
    end
    [train_num f_num] = size(X);
    y = (y + ones(train_num,1))/2 ;
    t = 0.5 ;
    c = 0.5 ;
    theta_orign = theta ;
    dot_sig_y = sigmod(X*theta) - y; % train_num*1
    D =  X' * dot_sig_y   ; % f_num * 1 
    D = D + lambda*theta ;
    cost_orign = calcCost_LR(theta,X,y);
    for j=1:MAX_SEARCH
        theta = theta + step*D ;
        cost = calcCost_LR(theta,X,y);
        if cost <= cost_orign + (theta-theta_orign)'*D*step*c
            break;
        end
        step = t*step;
    end
end

