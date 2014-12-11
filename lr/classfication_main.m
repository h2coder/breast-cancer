%main entry for classfication algorithm

ratio = 0.1:0.1:0.9;
train_curve = [];
test_curve = [];
for j = 1:length(ratio)
    [train_pt,test_pt] = classficaton_grad(ratio(j));
    train_curve = [train_curve;train_pt];
    test_curve = [test_curve;test_pt];
end

figure;
plot(train_curve(:,1),train_curve(:,2),'r');
hold on;
plot(test_curve(:,1),test_curve(:,2));
xlabel('data amount');
ylabel('cost');
hold off;

