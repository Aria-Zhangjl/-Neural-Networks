%�������ز�
%û�н�����֤
clear all

%�������ݼ��ͱ�ǩ
Images = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:60000);
D = Labels(1:60000);

%��ʼ��Ȩ��

rng(1)
W1 = rand(100,784);
rng(1)
W2 = rand(50,100);
rng(1)
W3 = rand(40,50);
rng(1)
W4 = rand(10,40);

%����3�ֵ�ѵ��
for epoch = 1:3
  epoch
  [W1, W2,W3,W4] = Train_Mnist1(W1,W2,W3,W4, X, D);
end

%�������������
save('Cross_weight1.mat');


% Test
%����������ݼ�
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
    x  = reshape(X(:, :, k),784,1);                % ����
    v1=W1*x;
    y1=ReLU(v1);
    v2=W2*y1;
    y2=ReLU(v2);
    v3=W3*y2;
    y3=ReLU(v3);
    v4=W4*y3;
    y4=Softmax(v4);

  [~, i] = max(y4);
  if i == D(k)                                     % ��֤���
    acc = acc + 1;
  end
end

acc = acc / N;                                     % �����ȷ��
fprintf('Accuracy is %f\n', acc);

