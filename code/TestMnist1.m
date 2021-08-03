%三个隐藏层
%没有交叉验证
clear all

%导入数据集和标签
Images = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:60000);
D = Labels(1:60000);

%初始化权重

rng(1)
W1 = rand(100,784);
rng(1)
W2 = rand(50,100);
rng(1)
W3 = rand(40,50);
rng(1)
W4 = rand(10,40);

%进行3轮的训练
for epoch = 1:3
  epoch
  [W1, W2,W3,W4] = Train_Mnist1(W1,W2,W3,W4, X, D);
end

%将结果保存下来
save('Cross_weight1.mat');


% Test
%导入测试数据集
Images = loadMNISTImages('D:\服装数据集\t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\服装数据集\t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:10000);
D = Labels(1:10000);

%开始测试
acc = 0;
N   = length(D);
for k = 1:N
    x  = reshape(X(:, :, k),784,1);                % 计算
    v1=W1*x;
    y1=ReLU(v1);
    v2=W2*y1;
    y2=ReLU(v2);
    v3=W3*y2;
    y3=ReLU(v3);
    v4=W4*y3;
    y4=Softmax(v4);

  [~, i] = max(y4);
  if i == D(k)                                     % 验证结果
    acc = acc + 1;
  end
end

acc = acc / N;                                     % 输出正确率
fprintf('Accuracy is %f\n', acc);

