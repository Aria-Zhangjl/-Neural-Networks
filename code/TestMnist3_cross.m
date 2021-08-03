%������֤
%�����������������

%���ݼ��ϲ�
Images1 = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X1 = Images1(:, :, 1:60000);
D1 = Labels1(1:60000);

Images2 = loadMNISTImages('D:\��װ���ݼ�\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\��װ���ݼ�\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X2 = Images2(:, :, 1:10000);
D2 = Labels2(1:10000);
X = cat(3,X1,X2);
D = [D1;D2];

Accuracy=0;

%����֮ǰ��Ȩֵ
load('Cross_weight3.mat')

%������֤����
indices = crossvalind('Kfold', 70000, 10);
for k = 1:10
    test = (indices == k); 
    train = ~test;
	
	%ȡ��ѵ������
    train_data=X(:,:,train);
    train_target=D(train,:);
    %����3��ѵ��
    for epoch = 1:3
      epoch
      [W1, W2,W3,W4,W5] = Train_Mnist3(W1,W2,W3,W4,W5, train_data, train_target);
    end
    
    %ȡ����������
    test_data=X(:,:,test);
    test_target=D(test,:);
    acc = 0;
    N   = length(test_target);
    for j = 1:N
      x  = test_data(:,:, j);        % 28x28������
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
      if i == test_target(j)
        acc = acc + 1;
      end
    end
  acc = acc / N;
  Accuracy=Accuracy+acc;
  fprintf('%d: Accuracy is %f\n', k,acc);  %���ÿ����֤��׼ȷ��
end

%����ƽ��׼ȷ�ʣ�������ģ��
Accuracy = Accuracy/10;
fprintf('Average accuracy is %f\n', Accuracy);
save('show3.mat');