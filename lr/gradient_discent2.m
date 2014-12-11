% the normal gradient discent algorithm

[train,test] = split_train_test('breast-cancer-wisconsin-tag.data');
fea_dim = 9 ;
[train_num fea_num] = size(train) ;
train_X = train(:,2:10);
train_X = [train_X ones(train_num,1)];
train_y = train(:,end);
options = optimset('GradObj', 'on', 'MaxIter', 100);
alpha = 0.01 ;
theta = zeros(fea_dim+1,1);
max_iters = 500 ;
%intial iteration
[cost,t] = costFunc(theta,alpha,train_X,train_y);
theta = t ;
fprintf('initial step %d jVal: %f \n',0,cost);
last_cost = cost ;

%while max_iters > 0 
cost_iter = zeros(1,max_iters) ;
for iter=1:max_iters
    [cost,t] = costFuncGrad2(theta,alpha,train_X,train_y);
    theta = t ;
    cost_diff = (last_cost - cost)/last_cost ;
    disp('----------theta after iter----------');
    disp(theta');
    fprintf('step %d jVal: %f  \n',iter,cost);
    %if cost_diff < 10*exp(-4) 
    %        break ;
    %end
    %max_iters = max_iters - 1 ;
    last_cost = cost ;
    cost_iter(iter) = last_cost;
end
% Create New Figure
figure; hold on;
plot(1:max_iters,cost_iter);

%[theta,cost] = fminunc(@(t)(costFunc(t,alpha,train_X,train_y)), theta, options);