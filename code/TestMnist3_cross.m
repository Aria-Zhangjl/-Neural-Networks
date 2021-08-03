%交叉验证
%两个卷积和两个隐藏

%数据集合并
Images1 = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X1 = Images1(:, :, 1:60000);
D1 = Labels1(1:60000);

Images2 = loadMNISTImages('D:\服装数据集\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\服装数据集\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X2 = Images2(:, :, 1:10000);
D2 = Labels2(1:10000);
X = cat(3,X1,X2);
D = [D1;D2];

Accuracy=0;

%导入之前的权值
load('Cross_weight3.mat')

%交叉验证分类
indices = crossvalind('Kfold', 70000, 10);
for k = 1:10
    test = (indices == k); 
    train = ~test;
	
	%取出训练数据
    train_data=X(:,:,train);
    train_target=D(train,:);
    %进行3轮训练
    for epoch = 1:3
      epoch
      [W1, W2,W3,W4,W5] = Train_Mnist3(W1,W2,W3,W4,W5, train_data, train_target);
    end
    
    %取出测试数据
    test_data=X(:,:,test);
    test_target=D(test,:);
    acc = 0;
    N   = length(test_target);
    for j = 1:N
      x  = test_data(:,:, j);        % 28x28的输入
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

      [~, i] = max(y);
      if i == test_target(j)
        acc = acc + 1;
      end
    end
  acc = acc / N;
  Accuracy=Accuracy+acc;
  fprintf('%d: Accuracy is %f\n', k,acc);  %输出每轮验证的准确率
end

%计算平均准确率，并保存模型
Accuracy = Accuracy/10;
fprintf('Average accuracy is %f\n', Accuracy);
save('show3.mat');