function [ jVal ] = calcCost_LR( theta,X,y )
%calc the jVal for the input data and model
%   Detailed explanation goes here
    [train_num f_num] = size(X);
    dot_theta_x = X * theta ;
    scala_y = dot_theta_x .* y ;
    exp_scala_y = exp(-scala_y);
    jVal = 0.0;
    for j=1:train_num
        if isinf(exp_scala_y(j))
            if y(j) == 1 
                jVal = jVal - dot_theta_x(j) ;
            else
                jVal = jVal + dot_theta_x(j) ;
            end
        else
            jVal = jVal + log(1+exp_scala_y(j));
        end
    end
end

