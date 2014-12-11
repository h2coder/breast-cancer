function [ auc ] = calcAuc( fp_tp )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    [row col] = size(fp_tp) ;
    last_tp = 0 ;
    last_fp = 0 ;
    auc = 0.0 ;
    for j=1:row
        fp = fp_tp(j,1);
        tp = fp_tp(j,2);
        auc = auc + (last_tp+tp)*(fp-last_fp)*0.5 ;
        last_fp = fp;
        last_tp = tp;
    end
end

