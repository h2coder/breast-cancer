% quasi-newton algorithm --logistic regression

[train,test] = split_train_test('breast-cancer-wisconsin-tag.data');
fea_dim = 9 ;
train_X = train(:,2:10);
train_X = [train_X ones(train_num,1)];
[train_num f_num] = size(train_X) ;
train_y = train(:,end);
options = optimset('GradObj', 'on', 'MaxIter', 100);
alpha = 0.0001 ;
L2_C = 0 ;
theta = zeros(fea_dim+1,1);
max_iters = 1000 ;
%intial iteration
[cost,t] = costFunc(theta,alpha,train_X,train_y,L2_C);
theta = t ;
fprintf('initial step %d jVal: %f \n',0,cost);
last_cost = cost ;

%while max_iters > 0 
cost_iter = zeros(1,max_iters) ;

%calc B_0
    %calc Hessian matrix
    dot_theta_x = train_X* theta ;
    sig_theta_x = sigmod(dot_theta_x);
    B = zeros(f_num);
    for k=1:f_num
        for s=1:f_num
            X_k = train_X(:,k);
            X_s = train_X(:,s);
            B(k,s) = sum((sig_theta_x.*(1-sig_theta_x)).*(X_k.*X_s));
        end
    end
 theta_last = theta;  
for iter=1:max_iters
    [cost,t,B] = costFuncDFP(theta,alpha,train_X,train_y,L2_C,theta_last,B);    
    cost_diff = (last_cost - cost)/last_cost ;
    %disp('----------theta after iter----------');
    %disp(theta');
    fprintf('step %d jVal: %f  \n',iter,cost);
    %if cost_diff < 10*exp(-4) 
    %        break ;
    %end
    %max_iters = max_iters - 1 ;
    last_cost = cost ;
    cost_iter(iter) = last_cost;
    theta_last = theta ;
    theta = t ;
end
% Create New Figure
figure; hold on;
plot(1:max_iters,cost_iter);


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

figure; hold on;
plot(auc_table(:,1),auc_table(:,2)),title('AUC PLOT');
figure; hold on;
plot(accu_table(:,1),accu_table(:,2)),title('AUCCRACY');
fprintf('auc: %f \n',calcAuc(auc_table));
%[tp_rate fp_rate accuracy] = model_comment(theta,test_X,test_y,threshold);
%fprintf('threshold %f tp_rate:%f fp_rate:%f accuracy:%f \n',threshold,tp_rate,fp_rate,accuracy);
%[theta,cost] = fminunc(@(t)(costFunc(t,alpha,train_X,train_y)), theta, options);