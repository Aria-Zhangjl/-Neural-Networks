%三个隐藏层的交叉验证

%原来的训练集
Images = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X1 = Images(:, :, 1:60000);
D1 = Labels(1:60000);

%原来的测试集
Images = loadMNISTImages('D:\服装数据集\t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\服装数据集\t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X2 = Images(:, :, 1:10000);
D2 = Labels(1:10000);


X = cat(3,X1,X2);
D = [D1;D2];

Accuracy=0;

%导入权重
load('Cross_weight1.mat')
% Learning

%交叉验证中的分类
indices = crossvalind('Kfold', 70000, 10);
for k=1:10
    test = (indices == k); 
    train = ~test;
	
	%取出训练数据
    train_data=X(:,:,train);
    train_target=D(train,:);
    
	%进行10轮训练
    for epoch = 1:10
        epoch
        [W1, W2,W3,W4] = Train_Mnist1(W1,W2,W3,W4, train_data, train_target);
    end
    
	%取出测试数据
    test_data=X(:,:,test);
    test_target=D(test,:);
	
	%开始测试
    acc = 0;
    N   = length(test_target);
    for k1 = 1:N
      x  = test_data(:,:, k1);                %计算
      x=reshape(x,784,1);
      v1=W1*x;
      y1=ReLU(v1);
      v2=W2*y1;
      y2=ReLU(v2);
      v3=W3*y2;
      y3=ReLU(v3);
      v4=W4*y3;
      y4=Softmax(v4);

      [~, i] = max(y4);                         %进行验证
      if i == test_target(k1) 
         acc = acc + 1;
      end
    end
  acc = acc / N;
  Accuracy=Accuracy+acc;
  fprintf( '%d :Accuracy is %f\n',k,acc);      %输出每一轮交叉验证的正确率
end

fprintf( 'Average accuracy is %f\n',Accuracy/10);   %输出平均正确率
save('show1.mat');