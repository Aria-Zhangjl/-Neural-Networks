function [W1, W2,W3,W4,W5] = Train_Mnist3(W1, W2,W3,W4,W5,X, D)
%�����������������ز��ѵ������
%

alpha = 0.01;    %ѧϰ��0.01
beta  = 0.95;    %����������0.95

momentum1 = zeros(size(W1));  %��ʼ������
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
momentum4 = zeros(size(W4));
momentum5 = zeros(size(W5));

N = length(D);

%С����ÿһ������
bsize = 100;  
%ÿ�ֿ�ʼ����ʼ��
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
    % ǰ���㷨����
    x  = X(:, :, k);               % 28x28������
    y1 = Conv(x, W1);              % �����1,�õ�26x26x1�����

    y2 = ReLU(y1);                 
    y3 = Conv(y2,W2);              % �����2,�õ�16x16x32�����
    y4 = ReLU(y3);
    y5 = Pool(y4);                 % �ػ���,�õ�8x8x32�����
    

    y6 = reshape(y5, [], 1);       % ת��2048*1����
    v3 = W3*y6;                    % ���ز�1
    y7 = ReLU(v3);                 % 
    v4 = W4*y7;                    % ���ز�2
    y8 = ReLU(v4);
    v  = W5*y8;                    % �����
    y  = Softmax(v);               %

    % 
    % ��ȡ��ǩ
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % ���򴫲��㷨��������������
    %
    e      = d - y;                   % ��������  
    delta  = e;
    
    e5     = W5' * delta;             % ���ز�2�����
    delta5 = (y8 > 0) .* e5;
    
    e4     = W4' * delta5;            % ���ز�1�����
    delta4 = (y7 > 0) .* e4;    

    e3     = W3' * delta4;            % �ػ�������
       
    e2     = reshape(e3, size(y5));
    
    e1 = zeros(size(y4));   
   
    Wn = ones(size(y4)) / (2*2);
    for c = 1:32                      % �ػ�����󴫵����
      e1(:, :, c) = kron(e2(:, :, c), ones([2 2])) .* Wn(:, :, c);
    end
    
    delta3 = (y4 > 0) .* e1;          % �����2������  
   
    delta1_x2 = zeros(size(W2));       
    for c = 1:32                      % �����2�Ĺ�������������
      delta1_x2(:, :, c) = conv2(y2(:, :), rot90(delta3(:, :, c),2), 'valid');
    end    
    
    e0=zeros(size(y2));             
	W_n=rot90(W2,2);
    for c=1:32                       % ��������2�ķ��򴫲����
        e0=e0+conv2(delta3(:,:,c),rot90(W_n(:,:,c),2),'full');
    end
    
    e0=(y2>0).*e0;                   % ��������1������
	
    delta1_x1=zeros(size(W1));       % �����1�Ĺ�������������
    delta1_x1(:,:,1)=conv2(x(:,:),rot90(e0(:,:),2),'valid');
    
    
    
    
    
    dW1 = dW1 + delta1_x1;           % ����������ƽ��ֵ���¾���
    dW2 = dW2 + delta1_x2;
    dW3 = dW3 + delta4*y6';
    dW4 = dW4 + delta5*y7';
    dW5 = dW5 + delta*y8';    
  end 
  
  % Update weights
  %
  dW1 = dW1 / bsize;                 % ����������ƽ��ֵ������Ȩ�ؾ���
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