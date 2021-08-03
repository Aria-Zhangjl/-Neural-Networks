function [W1, W2, W3,W4] = Train_Mnist(W1, W2, W3,W4, X, D)
%三个隐藏层的训练函数


alpha = 0.01;    %学习率0.01
beta  = 0.95;    %动量法常量为0.95

N = length(D);

momentum1 = zeros(size(W1));   %初始化动量
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
momentum4 = zeros(size(W4));


%小批量每一批的个数
bsize = 100;  
%每轮开始的起始点
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));                 % 增量归零
  dW2 = zeros(size(W2));
  dW3 = zeros(size(W3));
  dW4 = zeros(size(W4));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
    
	%前向算法传播
    x  = reshape(X(:,:, k),784,1);         % 转换成784×1的向量
    v1=W1*x;
    y1=ReLU(v1);                           % 第一个隐藏层
    v2=W2*y1;
    y2=ReLU(v2);                           % 第二个隐藏层
    v3=W3*y2;
    y3=ReLU(v3);                           % 第三个隐藏层
    v4=W4*y3;
    y4=Softmax(v4);                        % 输出层
    
    
   

    
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;      % 获取标签

    % 反向传播算法计算误差
    %
    e      = d - y4;                   % 输出层  
    delta  = e;

    e4     = W4' * delta;              % 第三个隐藏层
    delta4 = (y3 > 0) .* e4;
    
    e3     = W3' * delta4;             % 第二个隐藏层
    delta3 = (y2 > 0) .* e3;
    
    e2     = W2' * delta3;             % 第一个隐藏层
    delta2 = (y1 > 0) .* e2;
    
  
    dW4 = dW4+delta*y3';               %计算增量和
    dW3 = dW3+delta4*y2';
    dW2 = dW2+delta3*y1';
    dW1 = dW1+delta2*x';
  end 
  
  % Update weights
  %
  dW1 = dW1 / bsize;                  % 增量和平均值进行更新
  dW2 = dW2 / bsize;
  dW3 = dW3 / bsize;
  dW4 = dW4 / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum2 = alpha*dW2 + beta*momentum2;
  W2        = W2 + momentum2;
   
  momentum3 = alpha*dW3 + beta*momentum3;
  W3        = W3 + momentum3;  
  
  momentum4 = alpha*dW4 + beta*momentum4;
  W4        = W4 + momentum4;  
end

end