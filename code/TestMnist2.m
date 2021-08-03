%һ����������ز�
%û�н�����֤

%����ѵ�������ݺͱ�ǩ
Images1 = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X = Images1(:, :, 1:60000);
D = Labels1(1:60000);

%��ʼ��Ȩ�ؾ���
rng(1);
W1 = 1e-2*randn([9 9 20]);
rng(1);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
rng(1);
Wo = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);

%�������ֵ�ѵ��
for epoch = 1:3
    epoch
    [W1, W5,Wo] = Train_Mnist2(W1,W5,Wo, X,D);
end

%����ѵ���õ�ģ��
save('Cross_weight2.mat');

%������Լ����ݺͱ�ǩ
Images2 = loadMNISTImages('D:\��װ���ݼ�\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\��װ���ݼ�\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X = Images2(:, :, 1:10000);
D = Labels2(1:10000);

%���в���
acc = 0;
N   = length(D);
for j = 1:N
  x  = X(:,:, j);                % ����㣬�õ�28x28�ľ���
  y1 = Conv(x, W1);              % ����㣬�õ�20x20x20�ľ���
  y2 = ReLU(y1);                 %
  y3 = Pool(y2);                 % �ػ��㣬���10x10x20
  y4 = reshape(y3, [], 1);       %
  v5 = W5*y4;                    % 
  y5 = ReLU(v5);                 % ���ز㣬���100x1������
  v  = Wo*y5;                    % ����㣬���10x1������
  y  = Softmax(v);        
  [~, i] = max(y);               %������֤
  if i == D(j)
     acc = acc + 1;
  end
end
acc = acc / N;
fprintf( 'Accuracy is %f\n',acc);  %�����ȷ��
