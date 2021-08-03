%两个卷积和两个隐藏层，没有交叉验证


%导入训练集数据和标签
Images = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10
X = Images(:, :, 1:60000);
D = Labels(1:60000);

%初始化权重矩阵
%一个3*3的过滤器
rng(1);
W1 = 1e-2*randn([3 3 1]);
%32个11*11的过滤器
rng(1);
W2 = 1e-2*randn([11 11 32]);
%两个隐藏层之间的矩阵
rng(1);
W3 = (2*rand(1000, 2048) - 1) * sqrt(6) / sqrt(1000 + 2048);

rng(1);
W4 = (2*rand( 100,  1000) - 1) * sqrt(6) / sqrt( 100 +  1000);

rng(1);
W5 = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);


%进行3轮训练
for epoch = 1:3
  epoch
  [W1, W2,W3,W4,W5] = Train_Mnist3(W1,W2,W3,W4,W5, X, D);
end

%保存训练好的模型
save('Cross_weight3.mat');


%获取测试数据集
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
    x  = X(:, :, k);               % Input,           28x28
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
   if i == D(k)
     acc = acc + 1;
   end
end
acc = acc / N;
fprintf('Accuracy is %f\n', acc);  % 输出正确率

