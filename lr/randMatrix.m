function [ MM ] = randMatrix( M )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    index = randperm(length(M));
    MM = [];
    for j=1:length(M)
        MM = [ MM;M(index(j),:)];
    end
end

