function [W1, W2,W3,W4,W5] = Train_Mnist3(W1, W2,W3,W4,W5,X, D)
%两个卷积层和两个隐藏层的训练函数
%

alpha = 0.01;    %学习率0.01
beta  = 0.95;    %动量法常量0.95

momentum1 = zeros(size(W1));  %初始化动量
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
momentum4 = zeros(size(W4));
momentum5 = zeros(size(W5));

N = length(D);

%小批量每一个个数
bsize = 100;  
%每轮开始的起始点
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));
  dW2 = zeros(size(W2));
  dW3 = zeros(size(W3));
  dW4 = zeros(size(W4));
  dW5 = zeros(size(W5));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
    %
    % 前向算法传播
    x  = X(:, :, k);               % 28x28的输入
    y1 = Conv(x, W1);              % 卷积层1,得到26x26x1的输出

    y2 = ReLU(y1);                 
    y3 = Conv(y2,W2);              % 卷积层2,得到16x16x32的输出
    y4 = ReLU(y3);
    y5 = Pool(y4);                 % 池化层,得到8x8x32的输出
    

    y6 = reshape(y5, [], 1);       % 转成2048*1矩阵
    v3 = W3*y6;                    % 隐藏层1
    y7 = ReLU(v3);                 % 
    v4 = W4*y7;                    % 隐藏层2
    y8 = ReLU(v4);
    v  = W5*y8;                    % 输出层
    y  = Softmax(v);               %

    % 
    % 获取标签
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % 反向传播算法传递误差、计算增量
    %
    e      = d - y;                   % 输出层误差  
    delta  = e;
    
    e5     = W5' * delta;             % 隐藏层2的误差
    delta5 = (y8 > 0) .* e5;
    
    e4     = W4' * delta5;            % 隐藏层1的误差
    delta4 = (y7 > 0) .* e4;    

    e3     = W3' * delta4;            % 池化层的误差
       
    e2     = reshape(e3, size(y5));
    
    e1 = zeros(size(y4));   
   
    Wn = ones(size(y4)) / (2*2);
    for c = 1:32                      % 池化层向后传递误差
      e1(:, :, c) = kron(e2(:, :, c), ones([2 2])) .* Wn(:, :, c);
    end
    
    delta3 = (y4 > 0) .* e1;          % 卷积层2的增量  
   
    delta1_x2 = zeros(size(W2));       
    for c = 1:32                      % 卷积层2的过滤器矩阵增量
      delta1_x2(:, :, c) = conv2(y2(:, :), rot90(delta3(:, :, c),2), 'valid');
    end    
    
    e0=zeros(size(y2));             
	W_n=rot90(W2,2);
    for c=1:32                       % 计算卷积层2的反向传播误差
        e0=e0+conv2(delta3(:,:,c),rot90(W_n(:,:,c),2),'full');
    end
    
    e0=(y2>0).*e0;                   % 计算卷积层1的增量
	
    delta1_x1=zeros(size(W1));       % 卷积层1的过滤器矩阵增量
    delta1_x1(:,:,1)=conv2(x(:,:),rot90(e0(:,:),2),'valid');
    
    
    
    
    
    dW1 = dW1 + delta1_x1;           % 计算增量和平均值更新矩阵
    dW2 = dW2 + delta1_x2;
    dW3 = dW3 + delta4*y6';
    dW4 = dW4 + delta5*y7';
    dW5 = dW5 + delta*y8';    
  end 
  
  % Update weights
  %
  dW1 = dW1 / bsize;                 % 计算增量和平均值并更新权重矩阵
  dW2 = dW2 / bsize;
  dW3 = dW3 / bsize;
  dW4 = dW4 / bsize;
  dW5 = dW5 / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum2 = alpha*dW2 + beta*momentum2;
  W2        = W2 + momentum2;
  
  momentum3 = alpha*dW3 + beta*momentum3;
  W3        = W3 + momentum3;
  
  momentum4 = alpha*dW4 + beta*momentum4;
  W4        = W4 + momentum4;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
end

end