function [jVal, t] = costFuncNewton(theta,alpha,X,y,lambda)
%clac grdient using newton method
%   Detailed explanation goes here
    if nargin < 5
        lambda = 0.0 ;
    end
    [train_num f_num] = size(X);
    disp('--------input theta------------');
    disp(theta');
    dot_theta_x = X* theta ; % train_num * 1
    disp('--------dot_theta_x------------');
    disp(dot_theta_x(1:15)');
    %fprintf('dot_theta_x size : %d %d\n',size(dot_theta_x));
    scala_y = dot_theta_x .* y ; % train_num * 1
    disp('--------scala_y------------');
    disp(scala_y(1:15)');
    %disp('--------sig_y_theta_x------------');
    sig_y_theta_x = sigmod(scala_y) ; % train_num * 1
    %disp(sig_y_theta_x(1:15)');
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
    jVal = jVal + lambda*0.5*(theta'*theta) ; 
    %Òì³£ÈÝ´í
    % sig_y_theta_x = 0 y=1 jVal += dot_theta_x
    % sig_y_theta_x = 0 y=0 jVal -= dot_theta_x
    %jVal = sum(log(sig_y_theta_x.^-1)) ;
    %jVal = sum(log(1+exp(-scala_y)));
    %fprintf('in iteration jval: %f \n',jVal);
    
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
    dot_sig_y = sig_y_theta_x - ones(train_num,1); % train_num*1
    D =  X' * (dot_sig_y .* y)  ; % f_num * 1
    %H = pinv(sum((sig_theta_x.*(1-sig_theta_x))*(X'*X)));
    D = D + lambda*theta ;
    P = (-H)*D ; %newton direction
    t = theta + alpha * P ;
end

