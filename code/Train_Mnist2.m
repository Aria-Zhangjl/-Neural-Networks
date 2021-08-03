%一个卷积和一个隐藏训练函数
function [W1, W5, Wo] = Train_Mnist(W1, W5, Wo, X, D)
%
%

alpha = 0.01;      %学习率0.01
beta  = 0.95;      %动量法常量0.95

momentum1 = zeros(size(W1));   %初始化动量
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);

%小批量每一个个数
bsize = 100;  
%每轮开始的起始点
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));
  dW5 = zeros(size(W5));
  dWo = zeros(size(Wo));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
  
  
    %前向算法传播
    x  = X(:, :, k);               % 28x28的输入
    y1 = Conv(x, W1);              % 卷积层，得到20x20x20的输出
    y2 = ReLU(y1);                 
    y3 = Pool(y2);                 % 池化层，得到10x10x20的输出
    y4 = reshape(y3, [], 1);       % 隐藏层
    v5 = W5*y4;                     
    y5 = ReLU(v5);                 
    v  = Wo*y5;                     
    y  = Softmax(v);               % 输出层

    % 
    %获取标签
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % 反向传播算法传递误差与增量
    e      = d - y;                   % 输出层  
    delta  = e;

    e5     = Wo' * delta;             % 隐藏层
    delta5 = (y5 > 0) .* e5;

    e4     = W5' * delta5;            % 池化层
    
    e3     = reshape(e4, size(y3));

    e2 = zeros(size(y2));           
    W3 = ones(size(y2)) / (2*2);
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta2 = (y2 > 0) .* e2;          % 卷积层误差
  
    delta1_x = zeros(size(W1));       
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');   % 卷积层增量
    end
    
    dW1 = dW1 + delta1_x;            %计算权重矩阵的增量和
    dW5 = dW5 + delta5*y4';    
    dWo = dWo + delta *y5';
  end 
  
  
  dW1 = dW1 / bsize;                 %计算增量和平均值并更新权重矩阵
  dW5 = dW5 / bsize;
  dWo = dWo / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
  momentumo = alpha*dWo + beta*momentumo;
  Wo        = Wo + momentumo;  
end

end