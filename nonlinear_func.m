function [ y ] = nonlinear_func( x, type )
%% 'sigmoid', 'Relu', 'sgn', 'linear'
if strcmp(type, 'sigmoid')
    y = (1 + exp(-x)).^(-1);
elseif strcmp(type, 'Relu')
    y = ones(size(x));
    y(x < 0) = 0;
elseif strcmp(type, 'sgn')
    y = ones(size(x));
    y(x <= 0) = 0;
    y(x > 0) = 1;
elseif strcmp(type, 'linear')
    y = x;
end
end

