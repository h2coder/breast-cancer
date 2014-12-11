function [jVal, t,B ] = costFuncDFP(theta,alpha,X,y,lambda,theta_last,B)
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
    
    dot_sig_y = sig_theta_x - y; % train_num*1
    D =  X' * dot_sig_y   ; % f_num * 1 
    D = D + lambda*theta ;
    
    %calc Hessian matrix
    y_b = zeros(f_num,1);
    s_b = zeros(f_num,1);
    p_b = 1;
    if theta ~= theta_last
        dot_sig_y_last = sigmod(X*theta_last) - y ;
        D_last = X' * dot_sig_y_last + lambda*theta_last ; 
        s_b = theta - theta_last ;
        y_b = D - D_last ;
        p_b = 1/(y_b'*s_b);
    end
    
    B = (eye(f_num)-p_b*(y_b*s_b'))*B*(eye(f_num)-p_b*(s_b*y_b'))...
        + y_b*p_b*y_b' ;
    H = pinv(B);

    P = (-H)*D ; %newton direction
    t = theta + alpha * P ;
    
end

