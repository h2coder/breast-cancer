%positive y = 1
%negative y = 0

%for standard, matrix refered to 1 column of row vectors
%X matrix train_num*feature_num
%y label vector, size is train_num
function [jVal, t] = costFuncGrad4Linear(theta,alpha,X,y)
    
    [train_num f_num] = size(X);
    y = (y + ones(train_num,1))/2 ;
    disp(y(1:15)');
    disp('--------input theta------------');
    disp(theta');
    dot_theta_x = X* theta ; % train_num * 1
    disp('--------dot_theta_x------------');
    disp(dot_theta_x(1:15)');
    %fprintf('dot_theta_x size : %d %d\n',size(dot_theta_x));

    %disp(sig_y_theta_x(1:15)');
    jVal = sum(0.5 * (norm(dot_theta_x - y) .^ 2 ))/train_num;
    
    dot_sig_y = dot_theta_x - y; % train_num*1
    D =  X' * dot_sig_y   ; % f_num * 1 
    t = theta - alpha * D ;

  