clear
close all
clc
%% set parameter
layer_size = [1,5,1];
layer_num = size(layer_size,2);
eta = 0.2;
momen = 0.4;
max_iter = 10000;
type = 'sigmoid';   % 'sigmoid', 'Relu', 'sgn','linear'
output_type = 'linear';
%% prepare dataset
input = [0:0.01:3]';
output = target_func_1(input);
mean_input = mean(input);
mean_output = mean(output);
std_input = std(input);
std_output = std(output);
input = (input - mean_input) / std_input;
output = (output - mean_output) / std_output;
train_size = size(input,1);
batch_size = 5;
%% initialzie weight 
w = cell(1,layer_num - 1);
for i = 1:1:layer_num-1
    w{i} = randn(layer_size(i)+1,layer_size(i+1)) * 2;
end
%% start iteration, with batch as a process unit
[ w_final, err ] = train_my( input,output, w, ...
    batch_size, layer_size, eta, momen, max_iter,...
    type, output_type);

%% show result
figure,plot([1:1:size(err,2)],err)
figure,
plot(input,output,'r*');
hold on
y_eval = cell(1,layer_num);
w = w_final;
for i = 1:1:layer_num-1
    y_eval{i} = zeros(layer_size(i)+1,train_size);
end
i = layer_num-1;
y_eval{i+1} = zeros(layer_size(i+1),train_size);
y_eval{1} = [input';ones(1,train_size)];
for i = 2:1:layer_num-1
    temp = nonlinear_func([w{i-1}' * y_eval{i-1}],type);
    y_eval{i} = [temp;ones(1,train_size)];
end
i = layer_num-1;
y_eval{i+1} = nonlinear_func(w{i}' * y_eval{i},output_type);
err_final = sum((output' - y_eval{layer_num}).^2) / train_size;
plot(input,y_eval{i+1},'b');
legend('real','predict')

%% target function
function [y] = target_func_1(x)
y = -x.*(x.^2-3.2*x+1.7^2).*(x-3)/2;
end