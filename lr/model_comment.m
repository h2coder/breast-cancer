function [ tp_rate fp_rate accuracy] = model_comment( model,test_X,test_y,threshold )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[test_num fea_num] = size(test_X);
predict = sigmod(test_X * model) ;
%disp(predict(1:15)');
%fprintf('predict max: %f min: %f',max(predict),min(predict));
TP = 0;
FP = 0;
T = 0;
NN =0 ;
for j=1:test_num
    if test_y(j) == 1 
        T = T + 1 ;
    end
    if predict(j) >= threshold && test_y(j) == 1
        TP = TP + 1;
    end
    if predict(j) >= threshold && test_y(j) == -1
        FP = FP + 1 ;
    end
    if predict(j) < threshold && test_y(j) == -1
        NN = NN + 1;
    end
end

accuracy = (TP+NN)/test_num ;

tp_rate = TP/T ;
fp_rate = FP/(test_num-T) ;
%fprintf('real_true_rate:%f \n',T/test_num);
%fprintf('threshold %f tp_rate:%f fp_rate:%f accuracy:%f \n',threshold,TP/T,FP/(test_num-T),accuracy);

end

