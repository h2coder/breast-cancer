%for standard, matrix refered to 1 column of row vectors
%X matrix train_num*feature_num
%y label vector, size is train_num
function [jVal, t] = costFunc(theta,alpha,X,y)
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
                jVal = jVal + dot_theta_x(j) ;
            else
                jVal = jVal - dot_theta_x(j) ;
            end
        else
            jVal = jVal + log(1+exp_scala_y(j));
        end
    end
    %“Ï≥£»›¥Ì
    % sig_y_theta_x = 0 y=1 jVal += dot_theta_x
    % sig_y_theta_x = 0 y=0 jVal -= dot_theta_x
    %jVal = sum(log(sig_y_theta_x.^-1)) ;
    %jVal = sum(log(1+exp(-scala_y)));
    fprintf('in iteration jval: %f \n',jVal);
    dot_sig_y = sig_y_theta_x - ones(train_num,1); % train_num*1
    D =  X' * (dot_sig_y .* y)  ; % f_num * 1 
    t = theta - alpha * D ;

    
