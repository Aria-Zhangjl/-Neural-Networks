%�������ز�Ľ�����֤

%ԭ����ѵ����
Images = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X1 = Images(:, :, 1:60000);
D1 = Labels(1:60000);

%ԭ���Ĳ��Լ�
Images = loadMNISTImages('D:\��װ���ݼ�\t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\��װ���ݼ�\t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X2 = Images(:, :, 1:10000);
D2 = Labels(1:10000);


X = cat(3,X1,X2);
D = [D1;D2];

Accuracy=0;

%����Ȩ��
load('Cross_weight1.mat')
% Learning

%������֤�еķ���
indices = crossvalind('Kfold', 70000, 10);
for k=1:10
    test = (indices == k); 
    train = ~test;
	
	%ȡ��ѵ������
    train_data=X(:,:,train);
    train_target=D(train,:);
    
	%����10��ѵ��
    for epoch = 1:10
        epoch
        [W1, W2,W3,W4] = Train_Mnist1(W1,W2,W3,W4, train_data, train_target);
    end
    
	%ȡ����������
    test_data=X(:,:,test);
    test_target=D(test,:);
	
	%��ʼ����
    acc = 0;
    N   = length(test_target);
    for k1 = 1:N
      x  = test_data(:,:, k1);                %����
      x=reshape(x,784,1);
      v1=W1*x;
      y1=ReLU(v1);
      v2=W2*y1;
      y2=ReLU(v2);
      v3=W3*y2;
      y3=ReLU(v3);
      v4=W4*y3;
      y4=Softmax(v4);

      [~, i] = max(y4);                         %������֤
      if i == test_target(k1) 
         acc = acc + 1;
      end
    end
  acc = acc / N;
  Accuracy=Accuracy+acc;
  fprintf( '%d :Accuracy is %f\n',k,acc);      %���ÿһ�ֽ�����֤����ȷ��
end

fprintf( 'Average accuracy is %f\n',Accuracy/10);   %���ƽ����ȷ��
save('show1.mat');