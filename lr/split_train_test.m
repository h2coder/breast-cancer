%split the input data into train_data and test_data, the last attribute of
%output indicates the positive or negative
function [train_data,test_data] = split_train_test(csv)
    data = csvread(csv) ;
    [row,col] = size(data);
    train_size = floor(row*0.8) ;
    %test_size = row - train_size;
    %tag_class = @(x) if(x(end)==4)x(end)=1 else x(end)=-1;
    train_data = data(1:train_size,1:end);
    test_data = data(train_size+1:end,1:end) ;
        
    