%split the input data into train_data and test_data, the last attribute of
%output indicates the positive or negative
function [train_data,test_data] = split_train_test(csv,train_ratio)
    if nargin < 2
        train_ratio = 0.8 ;
    end
    data = csvread(csv) ;
    data = randMatrix(data);
    [row,col] = size(data);
    train_size = floor(row*train_ratio) ;
    %test_size = row - train_size;
    %tag_class = @(x) if(x(end)==4)x(end)=1 else x(end)=-1;
    train_data = data(1:train_size,1:end);
    test_data = data(train_size+1:end,1:end) ;
        
    