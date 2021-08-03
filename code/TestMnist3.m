%����������������ز㣬û�н�����֤


%����ѵ�������ݺͱ�ǩ
Images = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:60000);
D = Labels(1:60000);

%��ʼ��Ȩ�ؾ���
%һ��3*3�Ĺ�����
rng(1);
W1 = 1e-2*randn([3 3 1]);
%32��11*11�Ĺ�����
rng(1);
W2 = 1e-2*randn([11 11 32]);
%�������ز�֮��ľ���
rng(1);
W3 = (2*rand(1000, 2048) - 1) * sqrt(6) / sqrt(1000 + 2048);

rng(1);
W4 = (2*rand( 100,  1000) - 1) * sqrt(6) / sqrt( 100 +  1000);

rng(1);
W5 = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);


%����3��ѵ��
for epoch = 1:3
  epoch
  [W1, W2,W3,W4,W5] = Train_Mnist3(W1,W2,W3,W4,W5, X, D);
end

%����ѵ���õ�ģ��
save('Cross_weight3.mat');


%��ȡ�������ݼ�
Images = loadMNISTImages('D:\��װ���ݼ�\t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\��װ���ݼ�\t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:10000);
D = Labels(1:10000);

%��ʼ����
acc = 0;
N   = length(D);
for k = 1:N
    x  = X(:, :, k);               % Input,           28x28
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

   [~, i] = max(y);
   if i == D(k)
     acc = acc + 1;
   end
end
acc = acc / N;
fprintf('Accuracy is %f\n', acc);  % �����ȷ��

