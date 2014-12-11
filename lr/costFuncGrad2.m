%positive y = 1
%negative y = 0

%for standard, matrix refered to 1 column of row vectors
%X matrix train_num*feature_num
%y label vector, size is train_num
function [jVal, t] = costFuncGrad2(theta,alpha,X,y,lambda)
    if nargin < 5
        lambda = 0.0 ;
    end
    [train_num f_num] = size(X);
    y = (y + ones(train_num,1))/2 ;
    disp('--------input theta------------');
    disp(theta');
    dot_theta_x = X* theta ; % train_num * 1
    disp('--------dot_theta_x------------');
    disp(dot_theta_x(1:15)');
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
    t = theta - alpha * D ;
end

    
