function [jVal, t] = costFuncNewton2(theta,alpha,X,y,lambda)
% different newton method
%   Detailed explanation goes here
    if nargin < 5
        lambda = 0.0 ;
    end
    alpha = backtracking_line_search( 1.0,theta,X,y,lambda );
    fprintf('current search step is : %f \n',alpha);
    [train_num f_num] = size(X);
    y = (y + ones(train_num,1))/2 ;
    %disp('--------input theta------------');
    %disp(theta');
    dot_theta_x = X* theta ; % train_num * 1
    %disp('--------dot_theta_x------------');
    %disp(dot_theta_x(1:15)');
    %fprintf('dot_theta_x size : %d %d\n',size(dot_theta_x));

    %disp('--------sig_theta_x------------');
    sig_theta_x = sigmod(dot_theta_x) ; % train_num * 1
    %disp(sig_y_theta_x(1:15)');
    jVal = 0.0;
    for j=1:train_num
        if y(j) == 1
            exp_pos = exp(-dot_theta_x);
            if isinf(exp_pos(j))
                jVal = jVal - dot_theta_x(j) ;
            else
                jVal = jVal + log(1+exp_pos(j));
            end
        else
            exp_false = exp(dot_theta_x) ;
            if isinf(exp_false(j))
                jVal = jVal + dot_theta_x(j) ;
            else
                jVal = jVal + log(1+exp_false(j));
            end
        end
    end
    %calc Hessian matrix
    sig_theta_x = sigmod(dot_theta_x);
    B = zeros(f_num);
    for k=1:f_num
        for s=1:f_num
            X_k = X(:,k);
            X_s = X(:,s);
            B(k,s) = sum((sig_theta_x.*(1-sig_theta_x)).*(X_k.*X_s));
        end
    end
    H = pinv(B);

    dot_sig_y = sig_theta_x - y; % train_num*1
    D =  X' * dot_sig_y   ; % f_num * 1 
    D = D + lambda*theta ;
    P = (-H)*D ; %newton direction
    t = theta + alpha * P ;
end

