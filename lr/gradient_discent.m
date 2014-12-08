% the normal gradient discent algorithm

[train,test] = split_train_test('breast-cancer-wisconsin-tag.data');
fea_dim = 9 ;
train_X = train(:,2:10);
train_y = train(:,end);
options = optimset('GradObj', 'on', 'MaxIter', 100);
alpha = 0.5 ;
theta = zeros(fea_dim,1);
max_iters = 4 ;
%intial iteration
[cost,t] = costFunc(theta,alpha,train_X,train_y);
theta = t ;
fprintf('initial step %d jVal: %f \n',0,cost);
last_cost = cost ;
while max_iters > 0  
    [cost,t] = costFunc(theta,alpha,train_X,train_y);
    theta = t ;
    cost_diff = (last_cost - cost)/last_cost ;
     disp('----------theta after iter----------');
    disp(theta');
    fprintf('step %d jVal: %f  \n',max_iters,cost);
    if cost_diff < 10*exp(-4) 
            break ;
    end
    max_iters = max_iters - 1 ;
    last_cost = cost ;
end
%[theta,cost] = fminunc(@(t)(costFunc(t,alpha,train_X,train_y)), theta, options);