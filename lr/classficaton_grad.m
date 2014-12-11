
function [ train_curve, test_curve ] = classficaton_grad( train_ratio )
%learning_curve contains two curves for error trend on train data and test
% data along with the data amount
% Detailed explanation goes here

% the normal gradient discent algorithm --logistic regression

[train,test] = split_train_test('breast-cancer-wisconsin-tag.data',train_ratio);
fea_dim = 9 ;
[train_num fea_num] = size(train) ;
train_X = train(:,2:10);
train_X = [train_X ones(train_num,1)];
train_y = train(:,end);
options = optimset('GradObj', 'on', 'MaxIter', 100);
alpha = 0.0001 ;
L1_C = 1 ;
theta = zeros(fea_dim+1,1);
max_iters = 500 ;
%intial iteration
[cost,t] = costFunc(theta,alpha,train_X,train_y,L1_C);
theta = t ;
fprintf('initial step %d jVal: %f \n',0,cost);
last_cost = cost ;

%while max_iters > 0 
cost_iter = zeros(1,max_iters) ;
for iter=1:max_iters
    [cost,t] = costFunc(theta,alpha,train_X,train_y,L1_C);
    %[cost,t] = costFuncGrad2(theta,alpha,train_X,train_y);
    %[cost,t] = costFuncGrad4Linear(theta,alpha,train_X,train_y);
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
%figure; hold on;
%plot(1:max_iters,cost_iter);




%measure model theta on test data
[test_num fea_num] = size(test);
test_X = test(:,2:10);
test_X = [test_X ones(test_num,1)];
test_y = test(:,end);
threshold = 0.6 ;
threshold_arr = 1:-0.01:0.01 ;
auc_table = zeros(length(threshold_arr),2) ;
accu_table = zeros(length(threshold_arr),2);
for j=1:length(threshold_arr)
    [tp_rate fp_rate accuracy] = model_comment(theta,test_X,test_y,threshold_arr(j));
    auc_table(j,1) = fp_rate;
    auc_table(j,2) = tp_rate;
    accu_table(j,1) = threshold_arr(j) ;
    accu_table(j,2) = accuracy;
end

%figure; hold on;
%plot(auc_table(:,1),auc_table(:,2)),title('AUC PLOT');
%figure; hold on;
%plot(accu_table(:,1),accu_table(:,2)),title('AUCCRACY');
fprintf('auc: %f \n',calcAuc(auc_table));
%[tp_rate fp_rate accuracy] = model_comment(theta,test_X,test_y,threshold);
%fprintf('threshold %f tp_rate:%f fp_rate:%f accuracy:%f \n',threshold,tp_rate,fp_rate,accuracy);
%[theta,cost] = fminunc(@(t)(costFunc(t,alpha,train_X,train_y)), theta, options);


%fill learning curve,(train_ratio,train_cost,test_cost)
train_curve = [train_ratio,last_cost] ;
test_curve = [1-train_ratio,calcCost_LR(theta,test_X,test_y)];
end

