clear
close all
clc
%% set input size, hidden size and output size
layer_size = [9,3,1];
layer_num = size(layer_size,2);
eta = 0.2;
momen = 0.4;
error_target = 0;
max_iter = 200;
type = 'Relu';  % 'sigmoid', 'Relu', 'sgn','linear'
output_type = 'sgn';
%% prepare dataset
input = [0,0,0,1,0,1,1,0,1;...
    0,1,1,0,0,0,0,1,1;...
    1,0,1,1,0,1,0,0,0;...
    1,1,0,0,0,0,1,1,0;...
    0,1,1,0,1,1,0,0,0;...
    1,1,0,1,1,0,1,1,1;...
    0,0,0,1,1,0,1,1,0;...
    0,0,0,0,1,1,0,1,1];
output = [1;1;1;1;0;0;0;0];
train_size = size(input,1);
batch_size = 1;
%% initialzie weight 
w = cell(1,layer_num - 1);
for i = 1:1:layer_num-1
    w{i} = randn(layer_size(i)+1,layer_size(i+1)) * 2;
end
%% start iteration, with batch as a process unit
[ w_final, err ] = train_my( input,output, w, ...
    batch_size, layer_size, eta, momen, max_iter,...
    type, output_type, error_target);
%% show result
figure,plot([1:1:size(err,2)],err)
w = w_final;
y_eval = cell(1,layer_num);
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
accuracy = sum(y_eval{i+1} == output') / train_size;
figure,
plot([1:1:train_size],output,'r*');
hold on
plot([1:1:train_size],y_eval{i+1},'b');
legend('real','predict')
fprintf('[*] accuracy: %.2f, number: %d\n',accuracy,sum(y_eval{i+1} == output'));
