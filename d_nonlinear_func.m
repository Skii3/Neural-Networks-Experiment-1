function [ dy ] = d_nonlinear_func( y, type )
%% 'sigmoid', 'Relu', 'sgn', 'linear'
if strcmp(type, 'sigmoid')
    dy = y .* (1 - y);
elseif strcmp(type, 'Relu')
    dy = 1;
elseif strcmp(type, 'sgn')
    dy = 1;
elseif strcmp(type, 'linear')
    dy = 1;
end


end

