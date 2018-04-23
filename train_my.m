function [ w_final, err ] = train_my( input, output, w, batch_size, layer_size, eta, momen, max_iter,type, output_type)
% the main forward propagation and backward propagation
layer_num = size(w,2) + 1;
train_size = size(input,1);
%% start iteration, with batch as a process unit
iter = 0;
y = cell(1,layer_num);
delta = cell(1,layer_num);
dw = cell(1,layer_num - 1);
for i = 1:1:layer_num-1
    dw{i} = zeros(layer_size(i)+1,layer_size(i+1));
end
while(1)    
    last_w = w;
    index = randperm(train_size);
    input_batch = input(index(1:batch_size),:);
    output_batch = output(index(1:batch_size),:);
    %% create space for nodes in different layers
    i = 1;
    for i = 1:1:layer_num-1
        y{i} = zeros(layer_size(i)+1,batch_size);
    end
    i = layer_num-1;
    y{i+1} = zeros(layer_size(i+1),batch_size);
    %% forward propogation
    y{1} = [input_batch';ones(1,batch_size)];
    for i = 2:1:layer_num-1
        temp = nonlinear_func([w{i-1}' * y{i-1}],type);
        y{i} = [temp;ones(1,batch_size)];
    end
    i = layer_num-1;
    y{i+1} = nonlinear_func(w{i}' * y{i},output_type);
    %% backward propogation
    delta{layer_num} = (output_batch'-y{layer_num}).*d_nonlinear_func(y{layer_num},output_type);
    for i = layer_num-1:-1:1
        temp = d_nonlinear_func(y{i},type).*(w{i}*delta{i+1});
        delta{i} = temp(1:layer_size(i),:);
    end
    for i = 1:1:layer_num-1
        % update w
        temp = momen * dw{i} + (1-0)*eta * y{i} * delta{i+1}' / batch_size;
        w{i} = w{i} + temp;
        dw{i} = w{i} - last_w{i};
    end
    %% metric
    y_eval = y;
    y_eval{1} = [input';ones(1,train_size)];
    for i = 2:1:layer_num-1
        temp = nonlinear_func([w{i-1}' * y_eval{i-1}],type);
        y_eval{i} = [temp;ones(1,train_size)];
    end
    i = layer_num-1;
    y_eval{i+1} = nonlinear_func(w{i}' * y_eval{i},output_type);

    iter = iter + 1;
    err(iter) = sum((output' - y_eval{layer_num}).^2) / train_size;
    if iter == 1
        err_min = err(iter);
        w_final = w;
    elseif err_min > err(iter)
        err_min = err(iter);
        w_final = w;
    end
    fprintf('[*] Iteration: %d, L2 error: %.4f\n',iter,err(iter));
    if iter > max_iter
        break;
    end
end


end

